import { useRef, useState } from "react";
import {
  BookOpen,
  FileText,
  FileUp,
  LoaderCircle,
  Trash2,
} from "lucide-react";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { paperApi } from "../../api/client";
import type { Paper } from "../../types/api";

function paperTitle(paper: Paper): string {
  return paper.metadata.title ?? paper.processing.stored_filename;
}

export function PaperLibrary() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const [progress, setProgress] = useState<string | null>(null);

  const papersQuery = useQuery({
    queryKey: ["papers"],
    queryFn: paperApi.list,
  });

  const uploadPipeline = useMutation({
    mutationFn: async (file: File) => {
      setProgress("Uploading paper...");
      const uploadResponse = await paperApi.upload(file);

      setProgress("Processing text and metadata...");
      await paperApi.process(uploadResponse.paper.id);

      setProgress("Creating searchable embeddings...");
      await paperApi.index(uploadResponse.paper.id);
    },
    onSuccess: async () => {
      setProgress(null);
      await queryClient.invalidateQueries({ queryKey: ["papers"] });
    },
    onError: () => {
      setProgress(null);
    },
  });

  const deletePaper = useMutation({
    mutationFn: paperApi.delete,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["papers"] });
    },
  });

  function handleFileSelection(file: File | undefined) {
    if (!file) {
      return;
    }

    const isPdf =
      file.type === "application/pdf" ||
      file.name.toLowerCase().endsWith(".pdf");

    if (!isPdf) {
      alert("Please select a PDF research paper.");
      return;
    }

    uploadPipeline.mutate(file);
  }

  function handleDelete(paper: Paper) {
    const confirmed = window.confirm(
      `Delete "${paperTitle(paper)}"? This cannot be undone.`,
    );

    if (confirmed) {
      deletePaper.mutate(paper.id);
    }
  }

  const errorMessage =
    uploadPipeline.error instanceof Error
      ? uploadPipeline.error.message
      : null;

  return (
    <aside className="glass-panel animate-fade-in flex flex-col rounded-2xl p-6">
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400">
          <BookOpen size={20} />
        </div>
        <h2 className="text-lg font-semibold tracking-tight text-white">Research Library</h2>
      </div>

      <input
        ref={fileInputRef}
        className="hidden"
        type="file"
        accept=".pdf,application/pdf"
        onChange={(event) => {
          handleFileSelection(event.target.files?.[0]);
          event.target.value = "";
        }}
      />

      <button
        className="group relative flex w-full items-center justify-center gap-2 overflow-hidden rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 px-4 py-3.5 font-medium text-white shadow-lg shadow-indigo-500/20 transition-all hover:shadow-indigo-500/40 disabled:cursor-not-allowed disabled:opacity-50"
        disabled={uploadPipeline.isPending}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="absolute inset-0 bg-white/20 opacity-0 transition-opacity group-hover:opacity-100" />
        {uploadPipeline.isPending ? (
          <LoaderCircle className="animate-spin" size={18} />
        ) : (
          <FileUp size={18} className="transition-transform group-hover:-translate-y-0.5" />
        )}
        {uploadPipeline.isPending ? "Preparing paper..." : "Upload paper"}
      </button>

      {progress && (
        <p className="mt-4 animate-fade-in rounded-xl border border-indigo-500/20 bg-indigo-500/10 px-4 py-3 text-sm text-indigo-200">
          {progress}
        </p>
      )}

      {errorMessage && (
        <p className="mt-4 animate-fade-in rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {errorMessage}
        </p>
      )}

      <div className="mt-8 space-y-3">
        {papersQuery.isPending && (
          <p className="animate-pulse text-sm text-indigo-200/50">Loading your papers...</p>
        )}

        {papersQuery.isError && (
          <p className="text-sm text-red-400">
            Could not load your research library.
          </p>
        )}

        {papersQuery.data?.papers.map((paper) => (
          <article
            key={paper.id}
            className="group relative animate-slide-up overflow-hidden rounded-xl border border-white/10 bg-white/5 p-4 transition-all hover:bg-white/10"
          >
            <div className="flex gap-4">
              <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-400">
                <FileText size={16} />
              </div>

              <div className="min-w-0 flex-1">
                <h3 className="truncate font-medium text-slate-100 transition-colors group-hover:text-white">
                  {paperTitle(paper)}
                </h3>

                <p className="mt-1 text-xs text-slate-400">
                  {paper.metadata.authors.slice(0, 2).join(", ") ||
                    "Author unavailable"}
                  {paper.metadata.year ? ` · ${paper.metadata.year}` : ""}
                </p>

                <div className="mt-3 flex items-center gap-2">
                  <span className="inline-flex items-center rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-400">
                    {paper.processing.status}
                  </span>
                  <span className="text-[10px] text-slate-500">
                    {paper.processing.total_pages} pages
                  </span>
                </div>
              </div>

              <button
                aria-label={`Delete ${paperTitle(paper)}`}
                className="opacity-0 transition-all hover:text-red-400 focus:opacity-100 group-hover:opacity-100 disabled:cursor-not-allowed text-slate-500"
                disabled={deletePaper.isPending}
                onClick={() => handleDelete(paper)}
              >
                <Trash2 size={18} />
              </button>
            </div>
          </article>
        ))}

        {!papersQuery.isPending &&
          !papersQuery.isError &&
          papersQuery.data?.papers.length === 0 && (
            <div className="animate-fade-in flex flex-col items-center justify-center rounded-xl border border-dashed border-white/20 bg-white/5 py-10 px-5 text-center">
              <div className="mb-4 rounded-full bg-white/5 p-3 text-slate-400">
                <BookOpen size={24} />
              </div>
              <p className="font-medium text-slate-200">No papers yet</p>
              <p className="mt-2 max-w-[200px] text-xs text-slate-400 text-center">
                Upload a PDF to build your research library.
              </p>
            </div>
          )}
      </div>
    </aside>
  );
}