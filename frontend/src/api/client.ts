import type {
  AskQuestionRequest,
  ChatAnswer,
  PaperDeleteResponse,
  PaperIndexResponse,
  PaperListResponse,
  PaperResponse,
  PaperUploadResponse,
} from "../types/api";

import { supabase } from "../lib/supabase";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

class ApiError extends Error {
  public readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session) {
    throw new ApiError("Please sign in to continue.", 401);
  }

  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${session.access_token}`);

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);

    throw new ApiError(
      body?.detail ?? "Something went wrong. Please try again.",
      response.status,
    );
  }

  return response.json() as Promise<T>;
}

export const paperApi = {
  list(): Promise<PaperListResponse> {
    return request<PaperListResponse>("/papers");
  },

  upload(file: File): Promise<PaperUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    return request<PaperUploadResponse>("/papers", {
      method: "POST",
      body: formData,
    });
  },

  process(paperId: string): Promise<PaperResponse> {
    return request<PaperResponse>(`/papers/${paperId}/process`, {
      method: "POST",
    });
  },

  index(paperId: string): Promise<PaperIndexResponse> {
    return request<PaperIndexResponse>(`/papers/${paperId}/index`, {
      method: "POST",
    });
  },

  delete(paperId: string): Promise<PaperDeleteResponse> {
    return request<PaperDeleteResponse>(`/papers/${paperId}`, {
      method: "DELETE",
    });
  },
};

export const chatApi = {
  ask(question: AskQuestionRequest): Promise<ChatAnswer> {
    return request<ChatAnswer>("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(question),
    });
  },
};