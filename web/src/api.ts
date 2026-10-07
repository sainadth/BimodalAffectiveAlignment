export type SessionInfo = {
  consent_md: string; debrief_md: string; consent_version: string; consented: boolean;
  trials_total: number; trials_done: number; felt_options: string[]; manip_options: string[]; manip_check_text: string;
  scale: { min: number; max: number; low: string; high: string };
  dictation: boolean;
  trial_items: { key: string; text: string }[]; pets_items: { key: string; text: string }[];
  scripted: Record<string, string>;
};

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method, headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

export const api = {
  session: (tok: string) => req<SessionInfo>('GET', `/api/session/${tok}`),
  consent: (tok: string, version: string) => req<{ ok: boolean }>('POST', `/api/session/${tok}/consent`, { version }),
  analyze: (tok: string, t: number, text: string, frames: string[], input_method: string) =>
    req<{ ok: boolean; face_detected: boolean }>('POST', `/api/session/${tok}/trial/${t}/analyze`, {
      text, frames, input_method,
    }),
  transcribe: (tok: string, audio: string) =>
    req<{ text: string }>('POST', `/api/session/${tok}/transcribe`, { audio }),
  felt: (tok: string, t: number, felt: string) => req<{ reply: string }>('POST', `/api/session/${tok}/trial/${t}/felt`, { felt }),
  rate: (tok: string, t: number, ratings: Record<string, number>, manip_check: string) =>
    req<{ ok: boolean; trials_done: number; finished: boolean }>('POST', `/api/session/${tok}/trial/${t}/rate`, { ratings, manip_check }),
  pets: (tok: string, pets: Record<string, number>, funnel: string, free_text: string) =>
    req<{ ok: boolean }>('POST', `/api/session/${tok}/pets`, { pets, funnel, free_text }),
  withdraw: (tok: string) => req<{ ok: boolean; rows_removed: number }>('POST', `/api/session/${tok}/withdraw`),
};
