import { Container, getContainer } from "@cloudflare/containers";

/**
 * ML service on Cloudflare (App 3). The Python engine (../Dockerfile, FastAPI + scikit-learn) runs in a Cloudflare Container;
 * this Worker starts it on demand and forwards requests to it. Only the backend Worker calls it (service binding ML_SERVICE),
 * and the engine additionally verifies the signed gateway identity, so it cannot be used directly.
 *
 * One named instance on purpose: running jobs, datasets and quiz state live in that process's memory.
 */
interface Env {
  ML_CONTAINER: DurableObjectNamespace<MlContainer>;
  [secret: string]: unknown;
}

/** Everything the engine reads from its environment. Set with `npm run secrets` (see ../.env.cloud.example). */
const PASS_THROUGH = [
  "GATEWAY_SECRET", "STORE_SECRET", "STORE_URL", "ADMIN_EMAILS", "FRONTEND_ORIGINS", "PUBLIC_APP_URL",
  "UPI_ID", "UPI_PAYEE_NAME", "ADMIN_NOTIFY_EMAIL", "RESEND_API_KEY", "MAIL_FROM",
  "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN",
  "YOUTUBE_API_KEY", "YOUTUBE_API_KEYS",
  "GROQ_API_KEY", "GROQ_API_KEYS", "CEREBRAS_API_KEY", "CEREBRAS_API_KEYS", "GEMINI_API_KEY", "GEMINI_API_KEYS",
  "MISTRAL_API_KEY", "MISTRAL_API_KEYS", "OPENROUTER_API_KEY", "OPENROUTER_API_KEYS", "LLM_GROUPS",
  "JOB_WORKERS", "MAX_QUEUED_JOBS", "DATASET_TTL_SECONDS", "RETENTION_DAYS", "MIN_DATASET_ROWS", "MIN_DATASET_COLS", "MAX_UPLOAD_MB", "MAX_TRAIN_ROWS", "MAX_JOB_SECONDS", "REMIND_AFTER_SECONDS",
];

export class MlContainer extends Container<Env> {
  defaultPort = 7860;
  // Idle containers sleep (and stop billing); a run in progress keeps traffic flowing, so it stays awake.
  sleepAfter = "30m";

  constructor(ctx: DurableObjectState<{}>, env: Env) {
    super(ctx, env);
    const vars: Record<string, string> = { APP_ENV: "production", AUTH_MODE: "gateway", STORE_BACKEND: "remote" };
    for (const k of PASS_THROUGH) {
      const v = env[k];
      if (typeof v === "string" && v) vars[k] = v;
    }
    this.envVars = vars;
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return getContainer(env.ML_CONTAINER, "main").fetch(request);
  },
} satisfies ExportedHandler<Env>;
