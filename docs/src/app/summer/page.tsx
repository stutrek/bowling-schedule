'use client';

import {
    SummerScheduleProvider,
    useSummerSchedule,
} from './context/SummerScheduleContext';
import TabBar from '../components/TabBar';
import SummerControlBar from './components/SummerControlBar';
import SummerCostBreakdownTable from './components/SummerCostBreakdownTable';
import SummerMatchupsTable from './components/SummerMatchupsTable';
import SummerLaneCountsTable from './components/SummerLaneCountsTable';
import SummerSlotCountsTable from './components/SummerSlotCountsTable';
import SummerTimeGapsTable from './components/SummerTimeGapsTable';

import SummerSameLaneTable from './components/SummerSameLaneTable';
import SummerScheduleEditor from './components/SummerScheduleEditor';

function PageContent() {
    const { schedule, cost, analysis } = useSummerSchedule();
    const hasSchedule = schedule && cost && analysis;

    return (
        <div className="flex flex-col lg:flex-row min-h-screen">
            <div
                className={`flex-1 p-4 overflow-y-auto ${hasSchedule ? 'lg:pr-2' : ''}`}
            >
                <TabBar active="summer" />
                <div className="prose max-w-none">
                    <h1>Summer Bowling Schedule</h1>
                    <p>
                        Visualizer for summer league schedules. 12 teams, 10
                        weeks, 5 game slots per week (game 5 uses only lanes
                        3-4).{' '}
                        <a
                            href="https://github.com/stutrek/bowling-schedule"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            View on GitHub
                        </a>
                    </p>

                    <SummerControlBar />

                    {hasSchedule && (
                        <>
                            <SummerCostBreakdownTable />
                            <SummerMatchupsTable />
                            <SummerLaneCountsTable />
                            <SummerSameLaneTable />
                            <SummerSlotCountsTable />
                            <SummerTimeGapsTable />
                        </>
                    )}
                </div>
            </div>

            {hasSchedule && <SummerScheduleEditor />}
        </div>
    );
}

export default function SummerPage() {
    return (
        <SummerScheduleProvider>
            <PageContent />
        </SummerScheduleProvider>
    );
}
