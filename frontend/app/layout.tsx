import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'FaceTrace',
  description: 'AI-powered visual investigation and evidence verification',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#050505] text-[#F4F3EF] antialiased">
        {children}
      </body>
    </html>
  );
}