import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import {
  LoaderCircle,
  LogOut,
} from "lucide-react";

import { AuthPage } from "./features/auth/AuthPage";
import { ChatPanel } from "./features/chat/ChatPanel";
import { PaperLibrary } from "./features/library/PaperLibrary";
import { supabase } from "./lib/supabase";

function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [isLoadingSession, setIsLoadingSession] = useState(true);

  useEffect(() => {
    async function loadSession() {
      const { data } = await supabase.auth.getSession();
      setSession(data.session);
      setIsLoadingSession(false);
    }

    void loadSession();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setIsLoadingSession(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  if (isLoadingSession) {
    return (
      <main className="animated-bg flex min-h-screen items-center justify-center">
        <LoaderCircle className="animate-spin text-brand" size={40} />
      </main>
    );
  }

  if (!session) {
    return <AuthPage />;
  }

  return (
    <main className="animated-bg min-h-screen text-slate-100 selection:bg-brand/30">
      <header className="glass-panel sticky top-0 z-50 border-b-0 border-white/10 shadow-lg">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="flex h-10 w-10 overflow-hidden items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-[0_0_15px_rgba(99,102,241,0.5)]">
              <img src="/logo.png" alt="AI Research Copilot Logo" className="h-full w-full object-cover" />
            </div>

            <div>
              <h1 className="bg-gradient-to-r from-indigo-200 via-white to-indigo-200 bg-clip-text text-lg font-bold tracking-tight text-transparent">
                AI Research Copilot
              </h1>
              <p className="text-xs font-medium text-indigo-200/70">
                {session.user.email}
              </p>
            </div>
          </div>

          <button
            className="group flex items-center gap-2 rounded-xl border border-white/5 bg-white/5 px-4 py-2 text-sm font-medium text-slate-300 transition-all hover:bg-white/10 hover:text-white"
            onClick={() => void supabase.auth.signOut()}
          >
            <LogOut size={16} className="transition-transform group-hover:-translate-x-0.5" />
            Sign out
          </button>
        </div>
      </header>

      <section className="mx-auto grid max-w-7xl gap-8 px-6 py-10 lg:grid-cols-[340px_1fr]">
        <PaperLibrary />
        <ChatPanel />
      </section>
    </main>
  );
}

export default App;