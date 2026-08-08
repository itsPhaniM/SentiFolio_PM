/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        term: {
          bg: "#0a0e14",
          panel: "#0f141d",
          head: "#151c28",
          border: "#1e2733",
          borderlt: "#2b3745",
          text: "#c7d2de",
          muted: "#6b7885",
          amber: "#f5a623",
          amberdim: "#7a5f26",
          green: "#2ee6a0",
          red: "#ff5f6d",
          cyan: "#4cc9f0",
          blue: "#5b8def",
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
