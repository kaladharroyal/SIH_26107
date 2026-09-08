-- BIS AI Assistant: Unified Relational & Telemetry Schema
-- Holds structured corpus entities and live user interaction / telemetry logs.

CREATE TABLE IF NOT EXISTS standards (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    is_number       TEXT NOT NULL,          -- e.g. "IS 1786", "IS 1070"
    part            TEXT,                   -- e.g. "Part 1", NULL if no parts
    revision_year   TEXT NOT NULL,          -- e.g. "2008", "2023"
    title           TEXT,
    is_current      INTEGER DEFAULT 1,      -- 1 for current, 0 for superseded
    superseded_by   INTEGER REFERENCES standards(id),
    source_file     TEXT NOT NULL,
    source_url      TEXT,
    content_hash    TEXT,
    ingested_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (is_number, part, revision_year)
);

CREATE TABLE IF NOT EXISTS clauses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_id     INTEGER REFERENCES standards(id) ON DELETE CASCADE,
    clause_number   TEXT NOT NULL,          -- e.g. "4.2", "Table 1"
    clause_title    TEXT,
    text            TEXT NOT NULL,
    page_start      INTEGER,
    page_end        INTEGER,
    chunk_id        TEXT UNIQUE NOT NULL,   -- matches vector DB chunk_id
    embedding_model TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clauses_standard ON clauses(standard_id);
CREATE INDEX IF NOT EXISTS idx_clauses_chunk_id ON clauses(chunk_id);

-- Product -> Mandatory Standard Mapping
CREATE TABLE IF NOT EXISTS product_standard_map (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name    TEXT NOT NULL,
    standard_code   TEXT NOT NULL,
    hsn_code        TEXT,
    category        TEXT,
    mandatory       INTEGER NOT NULL DEFAULT 0, -- 1 = Mandatory under QCO/CRS, 0 = Voluntary
    scheme_name     TEXT,                       -- e.g. "Scheme-I (ISI Mark)", "CRS"
    source_url      TEXT,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_product_name ON product_standard_map(product_name);
CREATE INDEX IF NOT EXISTS idx_product_standard ON product_standard_map(standard_code);

-- BIS Recognized Testing Laboratories Directory
CREATE TABLE IF NOT EXISTS labs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lab_id          TEXT UNIQUE,
    name            TEXT NOT NULL,
    city            TEXT,
    state           TEXT,
    address         TEXT,
    recognized_scopes TEXT,                     -- JSON or comma-separated standards
    disciplines     TEXT,                       -- Electrical, Chemical, Civil, etc.
    contact_info    TEXT,
    source_url      TEXT,
    is_recognized   INTEGER DEFAULT 1,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_labs_state ON labs(state);
CREATE INDEX IF NOT EXISTS idx_labs_city ON labs(city);

-- Live Telemetry and Query Interaction Logs
CREATE TABLE IF NOT EXISTS query_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_query      TEXT NOT NULL,
    detected_language TEXT,
    intent          TEXT,
    retrieved_chunk_ids TEXT,                   -- JSON list of chunk_ids
    generated_answer TEXT,
    confidence_score REAL,
    retrieval_ms    REAL,
    generation_ms   REAL,
    total_ms        REAL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_logs_intent ON query_logs(intent);
CREATE INDEX IF NOT EXISTS idx_query_logs_time ON query_logs(created_at);

-- User Feedback Ratings on Query Answers
CREATE TABLE IF NOT EXISTS feedback_ratings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id        INTEGER REFERENCES query_logs(id),
    rating          INTEGER NOT NULL,           -- +1 = helpful, -1 = unhelpful
    feedback_notes  TEXT,
    submitted_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
