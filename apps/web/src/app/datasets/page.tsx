import { Database } from "lucide-react";

export default function DatasetsPage() {
  return (
    <div className="space-y-8 animate-in fade-in duration-500 h-full flex flex-col">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dataset Registry</h1>
        <p className="text-muted-foreground mt-2">Manage and monitor training datasets.</p>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center text-center p-8 bg-card border border-border rounded-xl border-dashed">
        <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4 text-muted-foreground">
          <Database size={32} />
        </div>
        <h2 className="text-xl font-semibold mb-2">Dataset Manager Not Connected</h2>
        <p className="text-muted-foreground max-w-md">
          The dataset management UI will be implemented in future phases. Currently, datasets are managed programmatically via the backend loaders.
        </p>
      </div>
    </div>
  );
}
