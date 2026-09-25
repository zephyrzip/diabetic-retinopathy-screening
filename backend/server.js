require('dotenv').config();

const { spawn } = require('node:child_process');
const path = require('node:path');
const cors = require('cors');
const cloudinary = require('cloudinary').v2;
const express = require('express');
const multer = require('multer');
const { Pool } = require('pg');

const app = express();
const PORT = Number(process.env.PORT || 8000);
const DATABASE_URL = process.env.DATABASE_URL;
const REPORTS_DATABASE_URL =
  process.env.DOCTOR_DATABASE_URL || process.env.Doctor_URL || DATABASE_URL;
const PYTHON_BIN = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'python' : 'python3');
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

if (!DATABASE_URL) throw new Error('DATABASE_URL must be configured.');
if (!REPORTS_DATABASE_URL) throw new Error('DOCTOR_DATABASE_URL or DATABASE_URL must be configured.');
if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) throw new Error('PORT must be a valid TCP port.');

function databaseOptions(connectionString) {
  const hostname = new URL(connectionString).hostname;
  const localDatabase = hostname === 'localhost' || hostname === '127.0.0.1';
  const sslDisabled = process.env.DATABASE_SSL === 'false' || localDatabase;
  return {
    connectionString,
    ...(sslDisabled ? {} : { ssl: { rejectUnauthorized: false } }),
  };
}

const scansPool = new Pool(databaseOptions(DATABASE_URL));
const reportsPool = new Pool(databaseOptions(REPORTS_DATABASE_URL));
scansPool.on('error', (error) => console.error('Unexpected patient database error:', error.message));
reportsPool.on('error', (error) => console.error('Unexpected reports database error:', error.message));

const allowedOrigins = new Set(
  (process.env.FRONTEND_URL || '')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean),
);
if (process.env.NODE_ENV !== 'production') {
  allowedOrigins.add('http://localhost:5173');
  allowedOrigins.add('http://127.0.0.1:5173');
}

app.use(cors({
  origin(origin, callback) {
    callback(null, !origin || allowedOrigins.has(origin));
  },
  methods: ['GET', 'POST', 'PATCH', 'OPTIONS'],
}));
app.use(express.json({ limit: '1mb' }));

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES },
  fileFilter(_req, file, callback) {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.mimetype)) {
      callback(new Error('Choose a JPEG, PNG, or WEBP retinal image.'));
      return;
    }
    callback(null, true);
  },
});

async function initializeDatabase() {
  await scansPool.query(`
    CREATE TABLE IF NOT EXISTS patient_scans (
      scan_id SERIAL PRIMARY KEY,
      patient_id VARCHAR(50) NOT NULL,
      patient_name VARCHAR(100) NOT NULL,
      age INT NOT NULL DEFAULT 0,
      gender VARCHAR(20) NOT NULL DEFAULT 'unknown',
      fundus_image_url TEXT NOT NULL,
      cloudinary_public_id VARCHAR(255) NOT NULL,
      technician_id VARCHAR(50) NOT NULL DEFAULT 'unassigned',
      notes TEXT,
      processing_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
      processing_error TEXT,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
  `);
  await scansPool.query(`
    ALTER TABLE patient_scans
      ADD COLUMN IF NOT EXISTS processing_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
      ADD COLUMN IF NOT EXISTS processing_error TEXT;
  `);
  await reportsPool.query(`
    CREATE TABLE IF NOT EXISTS doctor_reports (
      report_id SERIAL PRIMARY KEY,
      scan_id INT NOT NULL,
      patient_id VARCHAR(50) NOT NULL,
      report_image_url TEXT NOT NULL,
      cloudinary_public_id VARCHAR(255) NOT NULL,
      ai_predictions JSONB NOT NULL,
      doctor_notes TEXT,
      severity_level VARCHAR(30) NOT NULL,
      doctor_id VARCHAR(50) NOT NULL,
      review_status VARCHAR(20) NOT NULL DEFAULT 'PENDING_REVIEW',
      reviewed_at TIMESTAMP WITH TIME ZONE,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
  `);
  await reportsPool.query(`
    CREATE INDEX IF NOT EXISTS doctor_reports_scan_created_idx
      ON doctor_reports (scan_id, created_at DESC);
  `);
  await reportsPool.query(`
    ALTER TABLE doctor_reports
      ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) NOT NULL DEFAULT 'PENDING_REVIEW',
      ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP WITH TIME ZONE;
  `);
}

function validScreeningId(value) {
  if (!/^[1-9]\d*$/.test(value)) return null;
  const id = Number(value);
  return Number.isSafeInteger(id) ? id : null;
}

function screeningPayload(scan, report) {
  return {
    ...scan,
    screening_id: scan.scan_id,
    status: report ? 'COMPLETED' : scan.processing_status,
    processing_error: scan.processing_error,
    ai_grade: report?.ai_predictions?.grade ?? null,
    confidence: report?.ai_predictions?.confidence ?? null,
    clinical_decision: report?.ai_predictions?.clinical_decision ?? null,
    is_referable: report ? report.severity_level === 'REFER' : null,
    result_image_url: report?.report_image_url ?? null,
    doctor_notes: report?.doctor_notes ?? null,
    doctor_id: report?.doctor_id ?? null,
    review_status: report?.review_status ?? null,
    reviewed_at: report?.reviewed_at ?? null,
  };
}

function uploadToCloudinary(buffer) {
  return new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      { folder: 'telemed_screenings', resource_type: 'image' },
      (error, result) => {
        if (error) {
          reject(new Error('Cloud image storage upload failed.'));
          return;
        }
        if (!result?.secure_url || !result.public_id) {
          reject(new Error('Cloud image storage returned an invalid response.'));
          return;
        }
        resolve(result);
      },
    );
    stream.on('error', () => reject(new Error('Cloud image storage upload failed.')));
    stream.end(buffer);
  });
}

async function runPrediction(imageUrl) {
  return new Promise((resolve, reject) => {
    const worker = spawn(
      PYTHON_BIN,
      [path.join(__dirname, 'run_ml.py'), 'screening', imageUrl],
      { cwd: __dirname, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] },
    );
    let output = '';
    let errorOutput = '';

    worker.stdout.setEncoding('utf8');
    worker.stderr.setEncoding('utf8');
    worker.stdout.on('data', (chunk) => { output += chunk; });
    worker.stderr.on('data', (chunk) => { errorOutput += chunk; });
    worker.once('error', () => reject(new Error(`Could not start the screening worker (${PYTHON_BIN}).`)));
    worker.once('close', (code) => {
      if (code !== 0) {
        console.error('Screening worker failed:', errorOutput.slice(-2000));
        reject(new Error('The screening worker failed.'));
        return;
      }
      try {
        const result = JSON.parse(output.trim());
        if (
          !Number.isInteger(result.grade) || result.grade < 0 || result.grade > 4
          || typeof result.confidence !== 'number'
          || !Number.isFinite(result.confidence)
          || result.confidence < 0 || result.confidence > 1
        ) {
          reject(new Error('The screening worker returned an invalid prediction.'));
          return;
        }
        resolve(result);
      } catch (error) {
        reject(error instanceof SyntaxError
          ? new Error('The screening worker returned an invalid response.')
          : error);
      }
    });
  });
}

async function processScreening(scan) {
  try {
    const prediction = await runPrediction(scan.fundus_image_url);
    await reportsPool.query(
      `INSERT INTO doctor_reports
        (scan_id, patient_id, report_image_url, cloudinary_public_id, ai_predictions, severity_level, doctor_id)
       VALUES ($1, $2, $3, $4, $5, $6, $7)`,
      [
        scan.scan_id,
        scan.patient_id,
        scan.fundus_image_url,
        scan.cloudinary_public_id,
        JSON.stringify({ grade: prediction.grade, confidence: prediction.confidence }),
        prediction.grade >= 2 ? 'REFER' : 'NO_REFER',
        'pending-review',
      ],
    );
    await scansPool.query(
      'UPDATE patient_scans SET processing_status = $1, processing_error = NULL WHERE scan_id = $2',
      ['COMPLETED', scan.scan_id],
    );
  } catch (error) {
    console.error(`Screening ${scan.scan_id} failed:`, error.message);
    try {
      await scansPool.query(
        'UPDATE patient_scans SET processing_status = $1, processing_error = $2 WHERE scan_id = $3',
        ['FAILED', error.message, scan.scan_id],
      );
    } catch (updateError) {
      console.error(`Could not update status for screening ${scan.scan_id}:`, updateError.message);
    }
  }
}

app.get('/health', async (_req, res) => {
  try {
    await Promise.all([scansPool.query('SELECT 1'), reportsPool.query('SELECT 1')]);
    res.json({ status: 'ok' });
  } catch {
    res.status(503).json({ status: 'unavailable' });
  }
});

app.get('/api/screenings', async (_req, res, next) => {
  try {
    const { rows: scans } = await scansPool.query(
      'SELECT * FROM patient_scans ORDER BY created_at DESC, scan_id DESC LIMIT 100',
    );
    const scanIds = scans.map((scan) => scan.scan_id);
    const { rows: reports } = scanIds.length
      ? await reportsPool.query(
        `SELECT DISTINCT ON (scan_id) *
         FROM doctor_reports
         WHERE scan_id = ANY($1::int[])
         ORDER BY scan_id, created_at DESC`,
        [scanIds],
      )
      : { rows: [] };
    const reportsByScan = new Map(reports.map((report) => [report.scan_id, report]));
    res.json({ screenings: scans.map((scan) => screeningPayload(scan, reportsByScan.get(scan.scan_id))) });
  } catch (error) {
    next(error);
  }
});

app.post('/api/screenings/upload', upload.single('file'), async (req, res, next) => {
  if (!req.file) {
    res.status(400).json({ error: 'No image file provided.' });
    return;
  }
  try {
    const age = req.body.age === undefined || req.body.age === ''
      ? 0
      : Number.parseInt(req.body.age, 10);
    if (!Number.isInteger(age) || age < 0 || age > 120) {
      res.status(400).json({ error: 'Patient age must be between 0 and 120.' });
      return;
    }
    const image = await uploadToCloudinary(req.file.buffer);
    const { rows } = await scansPool.query(
      `INSERT INTO patient_scans
        (patient_id, patient_name, age, gender, fundus_image_url, cloudinary_public_id, technician_id)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING scan_id`,
      [
        String(req.body.patient_id || 'unassigned').slice(0, 50),
        String(req.body.patient_name || 'Unknown').slice(0, 100),
        age,
        String(req.body.gender || 'unknown').slice(0, 20),
        image.secure_url,
        image.public_id,
        String(req.body.technician_id || 'unassigned').slice(0, 50),
      ],
    );
    res.status(201).json({
      screening_id: rows[0].scan_id,
      image_url: image.secure_url,
      status: 'PENDING',
    });
  } catch (error) {
    next(error);
  }
});

app.post('/api/screenings/:id/process', async (req, res, next) => {
  const id = validScreeningId(req.params.id);
  if (!id) {
    res.status(400).json({ error: 'Screening ID must be a positive integer.' });
    return;
  }
  try {
    const { rows } = await scansPool.query(
      'SELECT * FROM patient_scans WHERE scan_id = $1',
      [id],
    );
    const scan = rows[0];
    if (!scan) {
      res.status(404).json({ error: 'Screening record not found.' });
      return;
    }
    if (scan.processing_status === 'PROCESSING') {
      res.status(409).json({ error: 'This screening is already processing.' });
      return;
    }
    if (scan.processing_status === 'COMPLETED') {
      res.status(409).json({ error: 'This screening has already completed.' });
      return;
    }
    await scansPool.query(
      'UPDATE patient_scans SET processing_status = $1, processing_error = NULL WHERE scan_id = $2',
      ['PROCESSING', id],
    );
    void processScreening(scan);
    res.status(202).json({ message: 'Screening processing started.', screening_id: id });
  } catch (error) {
    next(error);
  }
});

app.get('/api/screenings/:id', async (req, res, next) => {
  const id = validScreeningId(req.params.id);
  if (!id) {
    res.status(400).json({ error: 'Screening ID must be a positive integer.' });
    return;
  }
  try {
    const { rows: scanRows } = await scansPool.query(
      'SELECT * FROM patient_scans WHERE scan_id = $1',
      [id],
    );
    const scan = scanRows[0];
    if (!scan) {
      res.status(404).json({ error: 'Screening record not found.' });
      return;
    }
    const { rows: reports } = await reportsPool.query(
      'SELECT * FROM doctor_reports WHERE scan_id = $1 ORDER BY created_at DESC LIMIT 1',
      [id],
    );
    res.json(screeningPayload(scan, reports[0]));
  } catch (error) {
    next(error);
  }
});

app.patch('/api/screenings/:id/review', async (req, res, next) => {
  const id = validScreeningId(req.params.id);
  if (!id) {
    res.status(400).json({ error: 'Screening ID must be a positive integer.' });
    return;
  }

  const clinicalDecision = typeof req.body.clinical_decision === 'string'
    ? req.body.clinical_decision.trim()
    : '';
  const doctorNotes = typeof req.body.doctor_notes === 'string'
    ? req.body.doctor_notes.trim()
    : '';
  const doctorId = typeof req.body.doctor_id === 'string'
    ? req.body.doctor_id.trim()
    : '';
  if (!clinicalDecision || !doctorNotes || !doctorId) {
    res.status(400).json({ error: 'Clinical decision, notes, and doctor ID are required.' });
    return;
  }
  if (clinicalDecision.length > 100 || doctorNotes.length > 5000 || doctorId.length > 50) {
    res.status(400).json({ error: 'One or more review fields are too long.' });
    return;
  }

  try {
    const { rows } = await reportsPool.query(
      `UPDATE doctor_reports
       SET doctor_notes = $1,
           doctor_id = $2,
           review_status = 'SIGNED',
           reviewed_at = CURRENT_TIMESTAMP,
           ai_predictions = ai_predictions || $3::jsonb
       WHERE report_id = (
         SELECT report_id FROM doctor_reports
         WHERE scan_id = $4
         ORDER BY created_at DESC
         LIMIT 1
       )
       RETURNING *`,
      [doctorNotes, doctorId, JSON.stringify({ clinical_decision: clinicalDecision }), id],
    );
    if (!rows[0]) {
      res.status(409).json({ error: 'This screening does not have a completed AI report to review yet.' });
      return;
    }

    const { rows: scanRows } = await scansPool.query('SELECT * FROM patient_scans WHERE scan_id = $1', [id]);
    if (!scanRows[0]) {
      res.status(404).json({ error: 'Screening record not found.' });
      return;
    }
    res.json(screeningPayload(scanRows[0], rows[0]));
  } catch (error) {
    next(error);
  }
});

app.use((error, _req, res, _next) => {
  if (error instanceof multer.MulterError) {
    res.status(error.code === 'LIMIT_FILE_SIZE' ? 413 : 400).json({
      error: error.code === 'LIMIT_FILE_SIZE'
        ? 'Image must be 10 MB or smaller.'
        : 'The uploaded file could not be processed.',
    });
    return;
  }
  if (error.message === 'Choose a JPEG, PNG, or WEBP retinal image.') {
    res.status(415).json({ error: error.message });
    return;
  }
  console.error('API request failed:', error.message);
  res.status(500).json({ error: 'The screening service could not complete the request.' });
});

async function start() {
  await initializeDatabase();
  const server = app.listen(PORT, () => console.log(`Backend API listening on port ${PORT}.`));
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.once(signal, () => {
      server.close(async () => {
        await Promise.all([scansPool.end(), reportsPool.end()]);
        process.exit(0);
      });
    });
  }
}

start().catch(async (error) => {
  console.error('Backend startup failed:', error.message);
  await Promise.all([scansPool.end(), reportsPool.end()]);
  process.exitCode = 1;
});
