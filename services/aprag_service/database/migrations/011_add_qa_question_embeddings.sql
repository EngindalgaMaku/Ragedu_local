-- Migration: Add QA question embedding storage
-- This allows us to pre-compute and store embeddings for QA questions
-- Instead of fetching 50 QA pairs and calculating embeddings on-the-fly,
-- we can fetch QA pairs with their embeddings and do fast similarity search

-- Add columns to topic_qa_pairs table
-- SQLite doesn't support adding multiple columns in one ALTER TABLE statement
-- So we need to add them one by one

-- Add question_embedding column (JSON array of embedding values)
ALTER TABLE topic_qa_pairs ADD COLUMN question_embedding TEXT;

-- Add embedding_model column (Which embedding model was used)
ALTER TABLE topic_qa_pairs ADD COLUMN embedding_model VARCHAR(100);

-- Add embedding_dim column (Embedding dimension)
ALTER TABLE topic_qa_pairs ADD COLUMN embedding_dim INTEGER;

-- Add embedding_updated_at column (When embedding was last updated)
ALTER TABLE topic_qa_pairs ADD COLUMN embedding_updated_at TIMESTAMP;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_qa_pairs_topic_active ON topic_qa_pairs(topic_id, is_active);
CREATE INDEX IF NOT EXISTS idx_qa_pairs_embedding_model ON topic_qa_pairs(embedding_model);

-- Note: Existing QA pairs will have NULL embeddings
-- They will be computed on-demand when first accessed, or via a batch update script



