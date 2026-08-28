-- BIS Assistant — Phase 1, Step 4: structured storage schema
-- This mirrors what gets embedded into the vector DB, so you can always
-- trace a vector search hit back to its exact source clause, page, and PDF.

CREATE TABLE standards (
    id              SERIAL PRIMARY KEY,
    is_number       TEXT NOT NULL,          -- e.g. "IS 302"
    part            TEXT,                   -- e.g. "1", NULL if no parts
    revision_year   TEXT NOT NULL,
    title           TEXT,
    is_current      BOOLEAN DEFAULT TRUE,   -- flip to FALSE when superseded
    superseded_by   INTEGER REFERENCES standards(id),
    source_file     TEXT NOT NULL,
    source_url      TEXT,
    content_hash    TEXT NOT NULL,          -- from scraper manifest — detects revisions
    ingested_at     TIMESTAMPTZ DEFAULT now(),
    UNIQUE (is_number, part, revision_year)
);

CREATE TABLE clauses (
    id              SERIAL PRIMARY KEY,
    standard_id     INTEGER NOT NULL REFERENCES standards(id) ON DELETE CASCADE,
    clause_number   TEXT NOT NULL,          -- e.g. "4.2.1"
    clause_title    TEXT,
    text            TEXT NOT NULL,
    page_start      INTEGER,
    page_end        INTEGER,
    chunk_id        TEXT UNIQUE NOT NULL,   -- matches the vector DB record id
    embedding_model TEXT,                   -- track which model embedded this, for re-embed migrations
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_clauses_standard ON clauses(standard_id);
CREATE INDEX idx_clauses_text_search ON clauses USING GIN (to_tsvector('english', text));

-- Product -> mandatory standard mapping (CRS/QCO lists)
CREATE TABLE product_standard_map (
    id              SERIAL PRIMARY KEY,
    product_name    TEXT NOT NULL,
    hsn_code        TEXT,
    category        TEXT,
    standard_id     INTEGER REFERENCES standards(id),
    mandatory       BOOLEAN NOT NULL,       -- CRS/QCO = mandatory, others voluntary
    scheme_name     TEXT,                   -- e.g. "CRS", "ISI Mark", "Hallmarking"
    source_url      TEXT,
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_product_search ON product_standard_map USING GIN (to_tsvector('english', product_name));

-- BIS-recognized testing labs
CREATE TABLE labs (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    city            TEXT,
    state           TEXT,
    recognized_scopes TEXT[],               -- e.g. {'electrical', 'textiles'}
    contact_info    TEXT,
    source_url      TEXT,
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_labs_state ON labs(state);
CREATE INDEX idx_labs_scopes ON labs USING GIN (recognized_scopes);

-- Query logs for the feedback loop mentioned in Phase 6
CREATE TABLE query_logs (
    id              SERIAL PRIMARY KEY,
    user_query      TEXT NOT NULL,
    detected_language TEXT,
    intent          TEXT,                   -- 'general_qa' | 'product_lookup' | 'certification_guide' | 'lab_locator' | 'consumer_complaint'
    retrieved_chunk_ids TEXT[],
    generated_answer TEXT,
    confidence_score FLOAT,
    user_feedback   SMALLINT,               -- 1 = thumbs up, -1 = thumbs down, NULL = no feedback
    created_at      TIMESTAMPTZ DEFAULT now()
);
