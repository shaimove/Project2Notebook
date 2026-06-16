// Thin client for the Project2Notebook FastAPI backend.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface ProjectInfo {
  project_id: string;
  name: string;
  created_at: string;
  project_document_path?: string | null;
  csv_paths: string[];
  pdf_paths: string[];
  status: string;
}

export interface TimelineItem {
  step: number;
  title: string;
  status: string;
  detail: string;
  duration_ms?: number;
}

export interface ToolCall {
  server: string;
  tool: string;
  input: Record<string, unknown>;
  output_summary: string;
  status: string;
  duration_ms: number;
}

export interface RunResponse {
  project_id: string;
  status: string;
  timeline: TimelineItem[];
  tool_calls: ToolCall[];
  artifacts: Record<string, any>;
  summary: string;
  errors: PipelineError[];
}

export interface PipelineError {
  step: string;
  error: string;
}

async function jsonOrThrow(res: Response) {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export async function createProject(name: string): Promise<ProjectInfo> {
  const res = await fetch(`${API_BASE}/api/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return jsonOrThrow(res);
}

export async function uploadFile(
  projectId: string,
  kind: "project-document" | "csv" | "pdf",
  file: File
): Promise<any> {
  const form = new FormData();
  form.append("project_id", projectId);
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/upload/${kind}`, {
    method: "POST",
    body: form,
  });
  return jsonOrThrow(res);
}

export async function runPipeline(
  projectId: string,
  opts: { enablePriorArt: boolean; maxIterations: number; minRelImprovement: number }
): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_id: projectId,
      enable_prior_art: opts.enablePriorArt,
      max_iterations: opts.maxIterations,
      min_relative_improvement: opts.minRelImprovement,
    }),
  });
  return jsonOrThrow(res);
}

export async function getNotebook(projectId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/api/projects/${projectId}/notebook`);
  return jsonOrThrow(res);
}

export function plotUrl(projectId: string, name: string): string {
  return `${API_BASE}/api/projects/${projectId}/plots/${name}`;
}

export function notebookDownloadUrl(projectId: string): string {
  return `${API_BASE}/api/projects/${projectId}/notebook/download`;
}

export async function listTools(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/tools`);
  return jsonOrThrow(res);
}
