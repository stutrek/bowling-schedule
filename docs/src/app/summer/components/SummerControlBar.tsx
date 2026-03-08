'use client';

import { useSummerSchedule } from '../context/SummerScheduleContext';
import { summerScheduleToTSV } from '../lib/summer-schedule-utils';

export default function SummerControlBar() {
    const { schedule, resultFiles, importTSV, loadResultFile } =
        useSummerSchedule();

    async function pasteTSV() {
        try {
            const text = await navigator.clipboard.readText();
            if (!importTSV(text)) {
                alert('Could not parse summer TSV from clipboard');
            }
        } catch {
            alert('Could not read clipboard.');
        }
    }

    return (
        <div className="not-prose flex flex-col gap-3">
            <div className="flex items-center gap-4 flex-wrap">
                <select
                    defaultValue=""
                    onChange={(e) => {
                        const val = e.target.value;
                        if (val) loadResultFile(val);
                    }}
                    className="pl-3 pr-9 py-3 rounded-lg border border-gray-300 bg-white text-sm"
                >
                    <option value="" disabled>
                        Load a schedule...
                    </option>
                    {resultFiles.map((f) => (
                        <option key={f} value={f}>
                            {f.replace('.tsv', '')}
                        </option>
                    ))}
                </select>
                <button
                    type="button"
                    onClick={pasteTSV}
                    className="px-4 py-3 rounded-lg font-medium border border-gray-300 hover:bg-gray-100 transition-colors text-sm"
                >
                    Paste TSV
                </button>
                {schedule && (
                    <button
                        type="button"
                        onClick={() => {
                            navigator.clipboard.writeText(
                                summerScheduleToTSV(schedule),
                            );
                        }}
                        className="px-4 py-3 rounded-lg font-medium border border-gray-300 hover:bg-gray-100 transition-colors text-sm"
                    >
                        Copy TSV
                    </button>
                )}
            </div>
        </div>
    );
}
