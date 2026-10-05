export interface PaperMetadata {
  title: string | null;
  authors: string[];
  abstract: string | null;
  year: number | null;
  doi: string | null;
  journal: string | null;
  keywords: string[];
}

export interface PaperProcessing {
  stored_filename: string;
  total_pages: number;
  total_chunks: number;
  uploaded_at: string;
  status: string;
}

export interface Paper {
  id: string;
  metadata: PaperMetadata;
  processing: PaperProcessing;
}

export interface PaperUploadResponse {
  message: string;
  paper: Paper;
}

export interface PaperListResponse {
  papers: Paper[];
}

export interface PaperResponse {
  paper: Paper;
}

export interface PaperIndexResponse {
  message: string;
  indexed_chunks: number;
}

export interface PaperDeleteResponse {
  message: string;
}

export interface Citation {
  paper_id: string;
  paper_title: string;
  chunk_id: string;
  page_number: number;
}

export interface RetrievedChunk {
  chunk_id: string;
  score: number;
}

export interface ChatAnswer {
  answer_text: string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  confidence: number;
  generation_time: number;
}

export interface AskQuestionRequest {
  question: string;
  top_k: number;
  paper_id?: string;
}