import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        nature: {
          900: "#1a2f1a",
          800: "#2d4a2d",
          600: "#4f824f",
          500: "#609e60",
          200: "#c2e0c2",
          100: "#e0f0e0",
          50: "#f0f7f0",
        },
        earth: {
          50: "#efebe9",
        }
      },
    },
  },
  plugins: [],
};
export default config;