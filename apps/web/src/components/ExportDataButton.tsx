"use client";

import { Download } from "lucide-react";
import { API_BASE_URL } from "../lib/api";

export function ExportDataButton() {
  const handleExport = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/experiments/export/all`);
      const data = await res.json();
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement("a");
      a.href = url;
      a.download = `captcha_x_lab_export_${new Date().toISOString().split("T")[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed:", e);
      alert("Failed to export data.");
    }
  };

  return (
    <button 
      onClick={handleExport}
      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
    >
      <Download size={16} />
      <span>Export JSON</span>
    </button>
  );
}
