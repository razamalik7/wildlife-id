import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppProvider } from "@/lib/AppContext";
import BottomNav from "@/components/BottomNav";

export const viewport: Viewport = {
  themeColor: "#162A20",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export const metadata: Metadata = {
  title: "aniML vision — Field Press & Wildlife Collector Atlas",
  description:
    "Spot, scan, and collect North American wildlife — powered by OOGWAY computer vision.",
  manifest: "/manifest.json",
  icons: {
    icon: "/icons/icon-192.png",
    apple: "/icons/icon-192.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800;900&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@500;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased bg-[#EAE3D2] text-[#162A20]">
        <AppProvider>
          {children}
          <BottomNav />
        </AppProvider>
      </body>
    </html>
  );
}
