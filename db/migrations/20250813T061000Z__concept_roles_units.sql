-- Additional concept seeds for field roles and units
PRAGMA foreign_keys = ON;

-- Field roles used in spec resolution (value, date, etc.)
INSERT OR IGNORE INTO concept (type, key, name, description) VALUES
  ('field_role','value','Value','Numeric measure column'),
  ('field_role','date','Date','Date column'),
  ('field_role','person_ref','Person Reference','Person/employee identifier or name'),
  ('field_role','site_code','Site Code','Site/location identifier or name'),
  ('field_role','currency','Currency','Currency code or name'),
  ('field_role','supplier_code','Supplier Code','Supplier/vendor identifier or name'),
  ('field_role','unit','Unit','Unit column name');

INSERT OR IGNORE INTO concept_alias (concept_id, alias, lang, source, confidence)
SELECT c.id, a.alias, 'en', 'seed', 1.0
FROM concept c
JOIN (
  SELECT 'value' AS key, 'amount' AS alias UNION ALL
  SELECT 'value', 'total' UNION ALL
  SELECT 'value', 'sum' UNION ALL
  SELECT 'date', 'dt' UNION ALL
  SELECT 'date', 'day' UNION ALL
  SELECT 'person_ref', 'person' UNION ALL
  SELECT 'person_ref', 'employee' UNION ALL
  SELECT 'person_ref', 'name' UNION ALL
  SELECT 'site_code', 'site' UNION ALL
  SELECT 'site_code', 'location' UNION ALL
  SELECT 'currency', 'curr' UNION ALL
  SELECT 'currency', 'ccy' UNION ALL
  SELECT 'supplier_code', 'supplier' UNION ALL
  SELECT 'supplier_code', 'vendor' UNION ALL
  SELECT 'unit', 'units' UNION ALL
  SELECT 'unit', 'uom'
) a ON c.key = a.key AND c.type = 'field_role';

-- Units as concepts (for future: relations for conversions)
INSERT OR IGNORE INTO concept (type, key, name, description) VALUES
  ('unit','hours','Hours','Unit of time (hours)');

INSERT OR IGNORE INTO concept_alias (concept_id, alias, lang, source, confidence)
SELECT c.id, a.alias, 'en', 'seed', 1.0
FROM concept c
JOIN (
  SELECT 'hours' AS key, 'hrs' AS alias UNION ALL
  SELECT 'hours', 'hour' UNION ALL
  SELECT 'hours', 'h'
) a ON c.key = a.key AND c.type = 'unit';


