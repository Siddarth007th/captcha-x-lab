import React from "react";

interface FailureItem {
  expected: string;
  predicted: string;
  question?: string;
  question_type?: string;
}

export function FailureList({ failures }: { failures: FailureItem[] }) {
  if (!failures || failures.length === 0) {
    return (
      <div className="bg-card border border-border rounded-xl p-8 flex flex-col items-center justify-center shadow-sm border-dashed">
        <p className="text-muted-foreground text-center max-w-md">
          No failure examples recorded for this experiment run.
        </p>
      </div>
    );
  }

  // Basic categorization logic
  const categorized = failures.reduce((acc, curr) => {
    let category = "Misc Error";
    if (curr.predicted === "") {
      category = "Empty Prediction";
    } else if (curr.question_type) {
      category = `Type: ${curr.question_type}`;
    } else if (curr.expected.length !== curr.predicted.length) {
      category = "Length Mismatch";
    } else {
      category = "Character Substitution";
    }
    
    if (!acc[category]) acc[category] = [];
    acc[category].push(curr);
    return acc;
  }, {} as Record<string, FailureItem[]>);

  return (
    <div className="space-y-6">
      {Object.entries(categorized).map(([category, items]) => (
        <div key={category} className="bg-card border border-border rounded-xl shadow-sm overflow-hidden">
          <div className="bg-muted px-4 py-3 border-b border-border flex justify-between items-center">
            <h3 className="font-semibold text-sm uppercase text-foreground">{category}</h3>
            <span className="text-xs text-muted-foreground bg-background px-2 py-1 rounded-full border border-border">
              {items.length} cases
            </span>
          </div>
          <ul className="divide-y divide-border max-h-[300px] overflow-y-auto">
            {items.map((item, i) => (
              <li key={i} className="p-4 flex flex-col gap-1 text-sm hover:bg-muted/30 transition-colors">
                {item.question && (
                  <div className="text-xs text-muted-foreground mb-1 break-words">Q: {item.question}</div>
                )}
                <div className="flex gap-4">
                  <span className="text-emerald-500 font-mono break-all flex-1">Expected: {item.expected}</span>
                  <span className="text-rose-500 font-mono break-all flex-1">Predicted: {item.predicted || "<empty>"}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
