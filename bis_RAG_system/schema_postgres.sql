-- BIS AI Assistant: Unified PostgreSQL Relational & Telemetry Schema
-- Holds structured corpus entities and live user interaction / telemetry logs.

CREATE TABLE IF NOT EXISTS standards (
    id              BIGSERIAL PRIMARY KEY,
    is_number       VARCHAR(64) NOT NULL,          -- e.g. "IS 1786", "IS 1070"
    part            VARCHAR(64),                   -- e.g. "Part 1", NULL if no parts
    revision_year   VARCHAR(16) NOT NULL,          -- e.g. "2008", "2023"
    title           TEXT,
    is_current      SMALLINT DEFAULT 1,            -- 1 for current, 0 for superseded
    superseded_by   BIGINT REFERENCES standards(id),
    source_file     VARCHAR(512) NOT NULL,
    source_url      TEXT,
    content_hash    VARCHAR(128),
    ingested_at     TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (is_number, part, revision_year)
);

CREATE TABLE IF NOT EXISTS clauses (
    id              BIGSERIAL PRIMARY KEY,
    standard_id     BIGINT REFERENCES standards(id) ON DELETE CASCADE,
    clause_number   VARCHAR(64) NOT NULL,          -- e.g. "4.2", "Table 1"
    clause_title    TEXT,
    text            TEXT NOT NULL,
    page_start      INTEGER,
    page_end        INTEGER,
    chunk_id        VARCHAR(128) UNIQUE NOT NULL,  -- matches vector DB chunk_id
    embedding_model VARCHAR(128),
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clauses_standard ON clauses(standard_id);
CREATE INDEX IF NOT EXISTS idx_clauses_chunk_id ON clauses(chunk_id);

-- Product -> Mandatory Standard Mapping
CREATE TABLE IF NOT EXISTS product_standard_map (
    id              BIGSERIAL PRIMARY KEY,
    product_name    TEXT NOT NULL,
    standard_code   VARCHAR(128) NOT NULL,
    hsn_code        VARCHAR(64),
    category        VARCHAR(64),
    mandatory       SMALLINT NOT NULL DEFAULT 0,   -- 1 = Mandatory under QCO/CRS, 0 = Voluntary
    scheme_name     VARCHAR(128),                  -- e.g. "Scheme-I (ISI Mark)", "CRS"
    source_url      TEXT,
    updated_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_product_name ON product_standard_map(product_name);
CREATE INDEX IF NOT EXISTS idx_product_standard ON product_standard_map(standard_code);

-- BIS Recognized Testing Laboratories Directory
CREATE TABLE IF NOT EXISTS labs (
    id              BIGSERIAL PRIMARY KEY,
    lab_id          VARCHAR(128) UNIQUE,
    name            VARCHAR(256) NOT NULL,
    city            VARCHAR(128),
    state           VARCHAR(128),
    address         TEXT,
    recognized_scopes TEXT,                        -- JSON or comma-separated standards
    disciplines     TEXT,                          -- Electrical, Chemical, Civil, etc.
    contact_info    TEXT,
    source_url      TEXT,
    is_recognized   SMALLINT DEFAULT 1,
    updated_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_labs_state ON labs(state);
CREATE INDEX IF NOT EXISTS idx_labs_city ON labs(city);

-- Live Telemetry and Query Interaction Logs
CREATE TABLE IF NOT EXISTS query_logs (
    id              BIGSERIAL PRIMARY KEY,
    user_query      TEXT NOT NULL,
    detected_language VARCHAR(32),
    intent          VARCHAR(64),
    retrieved_chunk_ids TEXT,                      -- JSON array string of chunk_ids
    generated_answer TEXT,
    confidence_score DOUBLE PRECISION,
    retrieval_ms    DOUBLE PRECISION,
    generation_ms   DOUBLE PRECISION,
    total_ms        DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_logs_intent ON query_logs(intent);
CREATE INDEX IF NOT EXISTS idx_query_logs_time ON query_logs(created_at);

-- User Feedback Ratings on Query Answers
CREATE TABLE IF NOT EXISTS feedback_ratings (
    id              BIGSERIAL PRIMARY KEY,
    query_id        BIGINT REFERENCES query_logs(id) ON DELETE SET NULL,
    rating          INTEGER NOT NULL,              -- +1 = helpful, -1 = unhelpful
    feedback_notes  TEXT,
    submitted_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
