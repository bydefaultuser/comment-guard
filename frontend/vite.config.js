import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api calls are forwarded to the FastAPI backend, so the browser never needs the backend's port.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Allow links forwarded through VS Code's Ports panel (*.devtunnels.ms).
    allowedHosts: [".devtunnels.ms"],
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
