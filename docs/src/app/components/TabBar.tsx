'use client';

const tabs = [
    { label: 'Winter League', href: '' },
    { label: 'Summer League', href: '/summer' },
];

export default function TabBar({ active }: { active: 'winter' | 'summer' }) {
    const basePath =
        typeof window !== 'undefined'
            ? window.location.pathname.replace(/\/(summer\/?)?$/, '')
            : '/bowling-schedule';

    return (
        <div className="flex gap-1 mb-4">
            {tabs.map((tab) => {
                const isActive =
                    (active === 'winter' && tab.href === '') ||
                    (active === 'summer' && tab.href === '/summer');
                return (
                    <a
                        key={tab.href}
                        href={`${basePath}${tab.href}`}
                        className={`px-4 py-2 rounded-t-lg text-sm font-medium border border-b-0 transition-colors ${
                            isActive
                                ? 'bg-white text-blue-700 border-gray-300'
                                : 'bg-gray-100 text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-200'
                        }`}
                    >
                        {tab.label}
                    </a>
                );
            })}
        </div>
    );
}
