import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState } from "react";
import type { FormEvent } from "react";
import {
  Bot,
  LoaderCircle,
  Send,
  User,
} from "lucide-react";
import {
  useMutation,
  useQuery,
} from "@tanstack/react-query";

import { chatApi, paperApi } from "../../api/client";
import type {
  ChatAnswer,
  Citation,
} from "../../types/api";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  answer?: ChatAnswer;
}

function uniqueCitations(citations: Citation[]): Citation[] {
  return citations.filter(
    (citation, index, allCitations) =>
      allCitations.findIndex(
        (item) => item.chunk_id === citation.chunk_id,
      ) === index,
  );
}

function MarkdownAnswer({ content }: { content: string }) {
  return (
    <div className="markdown-answer">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}

export function ChatPanel() {
  const [question, setQuestion] = useState("");
  const [selectedPaperId, setSelectedPaperId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const papersQuery = useQuery({
    queryKey: ["papers"],
    queryFn: paperApi.list,
  });

  const askQuestion = useMutation({
    mutationFn: chatApi.ask,
    onSuccess: (answer) => {
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: answer.answer_text,
          answer,
        },
      ]);
    },
    onError: (error) => {
      const message =
        error instanceof Error
          ? error.message
          : "Could not answer your question.";

      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: message,
        },
      ]);
    },
  });

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const cleanQuestion = question.trim();

    if (!cleanQuestion || askQuestion.isPending) {
      return;
    }

    setMessages((currentMessages) => [
      ...currentMessages,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: cleanQuestion,
      },
    ]);

    setQuestion("");

    askQuestion.mutate({
      question: cleanQuestion,
      top_k: 5,
      ...(selectedPaperId && { paper_id: selectedPaperId }),
    });
  }

  const papers = papersQuery.data?.papers ?? [];
  const hasPapers = papers.length > 0;

  return (
    <section className="glass-panel animate-fade-in flex h-[calc(100vh-8rem)] min-h-[500px] flex-col overflow-hidden rounded-2xl">
      <div className="flex flex-col gap-4 border-b border-white/10 p-6 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white">Ask your papers</h2>
          <p className="mt-1 text-sm text-indigo-200/70">
            Answers are generated only from your private research library.
          </p>
        </div>

        <select
          className="glass-input cursor-pointer rounded-xl px-4 py-2.5 text-sm font-medium outline-none transition-all"
          value={selectedPaperId}
          onChange={(event) => setSelectedPaperId(event.target.value)}
        >
          <option value="" className="bg-slate-900 text-white">Search all my papers</option>

          {papers.map((paper) => (
            <option key={paper.id} value={paper.id} className="bg-slate-900 text-white">
              {paper.metadata.title ?? paper.processing.stored_filename}
            </option>
          ))}
        </select>
      </div>

      <div className="flex-1 space-y-8 overflow-y-auto p-6 custom-scrollbar">
        {!hasPapers && !papersQuery.isPending && (
          <div className="animate-fade-in flex h-full items-center justify-center text-center">
            <div className="max-w-md">
              <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-white/5 shadow-inner">
                <Bot className="text-indigo-400" size={40} />
              </div>
              <h3 className="text-xl font-semibold text-white">Upload a paper to begin</h3>
              <p className="mt-3 text-sm leading-relaxed text-indigo-200/70">
                Once uploaded, processed, and indexed, you can ask questions
                and receive answers with exact page citations.
              </p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <article
            key={message.id}
            className={`animate-slide-up flex gap-4 ${
              message.role === "user"
                ? "justify-end"
                : "justify-start"
            }`}
          >
            {message.role === "assistant" && (
              <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/20">
                <Bot size={20} />
              </div>
            )}

            <div
              className={`max-w-[85%] rounded-2xl px-5 py-4 shadow-sm ${
                message.role === "user"
                  ? "bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-indigo-500/20"
                  : "bg-white/10 backdrop-blur-md border border-white/10 text-slate-100"
              }`}
            >
              {message.role === "assistant" ? (
                <MarkdownAnswer content={message.text} />
                ) : (
                <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
                    {message.text}
                </p>
                )}

              {message.answer && (
                <div className="mt-5 border-t border-white/10 pt-4">
                  <p className="text-[11px] font-bold uppercase tracking-wider text-indigo-300">
                    Sources
                  </p>

                  <div className="mt-3 flex flex-wrap gap-2">
                    {uniqueCitations(message.answer.citations).map(
                      (citation) => (
                        <div
                          key={citation.chunk_id}
                          className="flex items-center gap-1.5 rounded-lg border border-white/5 bg-black/20 px-3 py-1.5 text-xs text-indigo-100"
                        >
                          <span className="font-medium">
                            {citation.paper_title}
                          </span>
                          <span className="text-indigo-400">·</span>
                          <span className="text-indigo-200">Page {citation.page_number}</span>
                        </div>
                      ),
                    )}
                  </div>

                  <div className="mt-4 flex items-center gap-2 text-[11px] text-slate-400">
                    <span>{message.answer.generation_time.toFixed(1)}s generation</span>
                  </div>
                </div>
              )}
            </div>

            {message.role === "user" && (
              <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/10 text-white">
                <User size={20} />
              </div>
            )}
          </article>
        ))}

        {askQuestion.isPending && (
          <div className="animate-slide-up flex gap-4">
            <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-[0_0_15px_rgba(99,102,241,0.5)]">
              <Bot size={20} className="animate-pulse" />
            </div>

            <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/10 px-5 py-4 text-[15px] text-slate-200 backdrop-blur-md">
              <LoaderCircle className="animate-spin text-indigo-400" size={18} />
              Reading your papers...
            </div>
          </div>
        )}
      </div>

      <form
        className="border-t border-white/10 bg-black/20 p-5 backdrop-blur-xl"
        onSubmit={handleSubmit}
      >
        <div className="flex gap-3">
          <input
            className="glass-input flex-1 rounded-xl px-5 py-4 text-[15px] outline-none transition-all disabled:opacity-50 placeholder-indigo-200/50"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={
              hasPapers
                ? "Ask a question about your research papers..."
                : "Upload a paper first..."
            }
            disabled={!hasPapers || askQuestion.isPending}
          />

          <button
            className="group relative flex items-center gap-2 overflow-hidden rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 px-6 py-4 font-medium text-white shadow-lg shadow-indigo-500/20 transition-all hover:shadow-indigo-500/40 disabled:cursor-not-allowed disabled:opacity-50"
            type="submit"
            disabled={
              !hasPapers ||
              !question.trim() ||
              askQuestion.isPending
            }
          >
            <div className="absolute inset-0 bg-white/20 opacity-0 transition-opacity group-hover:opacity-100" />
            <Send size={18} className="transition-transform group-hover:translate-x-1" />
            Ask
          </button>
        </div>
      </form>
    </section>
  );
}