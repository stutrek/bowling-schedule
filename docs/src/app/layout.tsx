import type { Metadata } from 'next';
// import logoIcon from '../../public/Logo Icon.png';
// import ogImage from '../../public/OG image.png';
import './globals.css';
import './markdown.css';
// import { Tracking } from './tracking';

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <head>
                {/* <link rel="icon" href={logoIcon.src} sizes="any" />
                <link rel="apple-touch-icon" href={logoIcon.src} sizes="any" /> */}
            </head>
            <body>
                {children}
                <div className="container max-w-3xl mx-auto p-4 text-sm text-gray-400">
                    <hr className="mb-3" />
                    &copy; 2024 Stu Kabakoff
                </div>
            </body>
            {/* <Tracking />
            <script
                defer
                async
                src="https://umami.stutrek.com/script.js"
                data-website-id=""
            /> */}
        </html>
    );
}

export const metadata: Metadata = {
    title: {
        template: '%s | Project',
        default: 'Bowling Schedule',
    },
    description: 'Attempts to make a more efficient bowling schedule',
    // openGraph: {
    //     images: [ogImage.src],
    // },
};
