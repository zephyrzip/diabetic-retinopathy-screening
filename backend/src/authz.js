export const ROLE_PERMISSIONS = Object.freeze({
  'Field technician': Object.freeze([
    'images:capture',
    'images:quality:grade',
    'images:submit',
  ]),
  Ophthalmologist: Object.freeze([
    'cases:read',
    'cases:referable:confirm',
    'evidence:gradcam:read',
  ]),
  'Program admin': Object.freeze([
    'program:caseload:read',
    'referrals:read',
    'referrals:manage',
    'screening-data:read',
    'screening-data:manage',
  ]),
});

export const API_PERMISSIONS = Object.freeze(
  [...new Set(Object.values(ROLE_PERMISSIONS).flat())],
);

// Auth0 must issue the permissions claim (API RBAC + "Add Permissions in the
// Access Token"). Missing claims fail closed.
export function requirePermissions(...requiredPermissions) {
  if (requiredPermissions.length === 0) {
    throw new Error('requirePermissions needs at least one permission.');
  }

  const unknownPermissions = requiredPermissions.filter(
    (permission) => !API_PERMISSIONS.includes(permission),
  );
  if (unknownPermissions.length > 0) {
    throw new Error(`Unknown API permission: ${unknownPermissions.join(', ')}`);
  }

  return (req, res, next) => {
    const granted = req.auth?.payload?.permissions;
    if (!Array.isArray(granted) || !requiredPermissions.every((p) => granted.includes(p))) {
      return res.status(403).json({ error: 'insufficient_permissions' });
    }
    return next();
  };
}
