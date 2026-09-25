export default function GradCAMViewer({ image }) {
  return <section><h2>Grad-CAM</h2>{image ? <img src={image} alt="Grad-CAM explanation"/> : <p>Grad-CAM not available yet.</p>}</section>;
}
