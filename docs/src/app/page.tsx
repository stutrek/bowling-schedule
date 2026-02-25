'use client';

import { ScheduleProvider, useSchedule } from './context/ScheduleContext';
import ControlBar from './components/ControlBar';
import SolverProgress from './components/SolverProgress';
import CostBreakdownTable from './components/CostBreakdownTable';
import MatchupsTable from './components/MatchupsTable';
import LaneCountsTable from './components/LaneCountsTable';
import LastGameLaneTable from './components/LastGameLaneTable';
import LaneSwitchesTable from './components/LaneSwitchesTable';
import SlotCountsTable from './components/SlotCountsTable';
import EarlyLateTable from './components/EarlyLateTable';
import ScheduleEditor from './components/ScheduleEditor';

function PageContent() {
    const { schedule, cost, analysis, violations } = useSchedule();
    const hasSchedule = schedule && cost && analysis && violations;

    return (
        <div className="flex flex-col lg:flex-row min-h-screen">
            <div
                className={`flex-1 p-4 overflow-y-auto ${hasSchedule ? 'lg:pr-2' : ''}`}
            >
                <div className="prose max-w-none">
                    <h1>Bowling Schedule Generator</h1>
                    <p>
                        Generates an optimized 12-week schedule for 16 teams
                        across 4 lanes. Teams stay on their lane pair (1-2 or
                        3-4), play every other team, and alternate early/late as
                        much as possible.
                    </p>

                    <ControlBar />
                    <SolverProgress />

                    {hasSchedule && (
                        <>
                            <CostBreakdownTable />
                            <MatchupsTable />
                            <LaneCountsTable />
                            <LastGameLaneTable />
                            <LaneSwitchesTable />
                            <SlotCountsTable />
                            <EarlyLateTable />
                        </>
                    )}
                </div>
            </div>

            {hasSchedule && <ScheduleEditor />}
        </div>
    );
}

export default function Page() {
    return (
        <ScheduleProvider>
            <PageContent />
        </ScheduleProvider>
    );
}
