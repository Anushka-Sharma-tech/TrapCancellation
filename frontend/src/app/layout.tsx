import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TrapCancellation | Bharat Voice Fraud Shield",
  description:
    "AI-powered real-time voice integrity verification for banks, fintech, telecom, and enterprise fraud defense",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}