import { useState } from "react";
import type { FormEvent } from "react";
import { LoaderCircle } from "lucide-react";

import { supabase } from "../../lib/supabase";

type AuthMode = "sign-in" | "sign-up";

export function AuthPage() {
  const [mode, setMode] = useState<AuthMode>("sign-in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setMessage(null);
    setErrorMessage(null);
    setIsSubmitting(true);

    try {
      if (mode === "sign-up") {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: window.location.origin,
          },
        });

        if (error) {
          throw error;
        }

        setMessage(
          "Account created. Check your email to confirm your account.",
        );
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (error) {
          throw error;
        }
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  const isSignUp = mode === "sign-up";

  return (
    <main className="animated-bg flex min-h-screen items-center justify-center px-6 py-12 selection:bg-brand/30">
      <section className="glass-panel animate-fade-in w-full max-w-md rounded-2xl p-8 shadow-2xl">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-6 flex h-14 w-14 overflow-hidden items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-[0_0_20px_rgba(99,102,241,0.5)]">
            <img src="/logo.png" alt="AI Research Copilot Logo" className="h-full w-full object-cover" />
          </div>

          <h1 className="bg-gradient-to-r from-indigo-200 via-white to-indigo-200 bg-clip-text text-2xl font-bold tracking-tight text-transparent">
            AI Research Copilot
          </h1>

          <p className="mt-3 text-sm text-indigo-200/70">
            {isSignUp
              ? "Create a private workspace for your research papers."
              : "Sign in to access your private research library."}
          </p>
        </div>

        <form className="space-y-5" onSubmit={handleSubmit}>
          <label className="block">
            <span className="mb-2 block text-sm font-medium text-slate-200">Email</span>
            <input
              className="glass-input w-full rounded-xl px-4 py-3.5 text-sm outline-none transition-all placeholder-indigo-200/50"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
              required
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm font-medium text-slate-200">Password</span>
            <input
              className="glass-input w-full rounded-xl px-4 py-3.5 text-sm outline-none transition-all placeholder-indigo-200/50"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="At least 8 characters"
              autoComplete={isSignUp ? "new-password" : "current-password"}
              minLength={8}
              required
            />
          </label>

          {errorMessage && (
            <p className="animate-fade-in rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">
              {errorMessage}
            </p>
          )}

          {message && (
            <p className="animate-fade-in rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
              {message}
            </p>
          )}

          <button
            className="group relative mt-2 flex w-full items-center justify-center gap-2 overflow-hidden rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 px-4 py-3.5 font-medium text-white shadow-lg shadow-indigo-500/20 transition-all hover:shadow-indigo-500/40 disabled:cursor-not-allowed disabled:opacity-50"
            type="submit"
            disabled={isSubmitting}
          >
            <div className="absolute inset-0 bg-white/20 opacity-0 transition-opacity group-hover:opacity-100" />
            {isSubmitting && <LoaderCircle className="animate-spin" size={18} />}
            {isSignUp ? "Create account" : "Sign in"}
          </button>
        </form>

        <button
          className="mt-6 block w-full text-center text-sm font-medium text-indigo-300 transition-colors hover:text-indigo-200"
          type="button"
          onClick={() => {
            setMode(isSignUp ? "sign-in" : "sign-up");
            setErrorMessage(null);
            setMessage(null);
          }}
        >
          {isSignUp
            ? "Already have an account? Sign in"
            : "New here? Create an account"}
        </button>
      </section>
    </main>
  );
}