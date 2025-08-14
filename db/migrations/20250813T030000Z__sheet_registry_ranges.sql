-- Extend sheet_registry with data_range, header_range, has_hidden
PRAGMA foreign_keys = ON;

ALTER TABLE sheet_registry ADD COLUMN data_range TEXT;
ALTER TABLE sheet_registry ADD COLUMN header_range TEXT;
ALTER TABLE sheet_registry ADD COLUMN has_hidden INTEGER DEFAULT 0;


