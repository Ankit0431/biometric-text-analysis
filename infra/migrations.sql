-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users (updated for authentication)
CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name TEXT NOT NULL,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  locale TEXT DEFAULT 'en',
  consent_version TEXT,
  biometric_enrolled BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT now(),
  deleted_at TIMESTAMPTZ
);

-- Profiles
CREATE TABLE IF NOT EXISTS profiles (
  user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
  lang TEXT NOT NULL,
  domain TEXT NOT NULL,
  centroid VECTOR(512) NOT NULL,
  cov_diag REAL[] NOT NULL,
  n_samples INT NOT NULL,
  stylometry_stats JSONB NOT NULL,
  threshold_high REAL NOT NULL,
  threshold_med REAL NOT NULL,
  prompt_answers JSONB, -- Store prompt_id -> answer_embedding mappings for semantic verification
  last_update TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (user_id, lang, domain)
);

-- Decisions
CREATE TABLE IF NOT EXISTS decisions (
  id BIGSERIAL PRIMARY KEY,
  user_id TEXT,
  kind TEXT,
  lang TEXT,
  domain TEXT,
  score REAL,
  decision TEXT,
  reasons JSONB,
  len_words INT,
  policy_version TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Cohort vectors (optional)
CREATE TABLE IF NOT EXISTS cohort_vectors (
  user_id TEXT,
  lang TEXT,
  domain TEXT,
  vec VECTOR(512),
  source_user_id TEXT
);

-- Index for approximate nearest neighbor on vec using hnsw
CREATE INDEX IF NOT EXISTS idx_cohort_vectors_vec ON cohort_vectors USING hnsw (vec vector_l2_ops);
