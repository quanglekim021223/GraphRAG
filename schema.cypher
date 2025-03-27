// Healthcare GraphRAG Database Schema
// ============================
//
// File này chứa các lệnh Cypher để khởi tạo schema cho cơ sở dữ liệu Neo4j Healthcare GraphRAG
// Bao gồm: constraints, indexes, và mô tả cấu trúc graph data model

// ========== CONSTRAINTS ==========
// Constraints đảm bảo tính toàn vẹn và duy nhất của dữ liệu

// Patient constraints
CREATE CONSTRAINT patient_name_constraint IF NOT EXISTS 
FOR (p:Patient) REQUIRE p.name IS UNIQUE;

// Disease constraints
CREATE CONSTRAINT disease_name_constraint IF NOT EXISTS 
FOR (d:Disease) REQUIRE d.name IS UNIQUE;

// Doctor constraints
CREATE CONSTRAINT doctor_name_constraint IF NOT EXISTS 
FOR (d:Doctor) REQUIRE d.name IS UNIQUE;

// Hospital constraints
CREATE CONSTRAINT hospital_name_constraint IF NOT EXISTS 
FOR (h:Hospital) REQUIRE h.name IS UNIQUE;

// InsuranceProvider constraints
CREATE CONSTRAINT insurance_name_constraint IF NOT EXISTS 
FOR (i:InsuranceProvider) REQUIRE i.name IS UNIQUE;

// Room constraints
CREATE CONSTRAINT room_number_constraint IF NOT EXISTS 
FOR (r:Room) REQUIRE r.room_number IS UNIQUE;

// Medication constraints
CREATE CONSTRAINT medication_name_constraint IF NOT EXISTS 
FOR (m:Medication) REQUIRE m.name IS UNIQUE;

// TestResults constraints - không cần unique constraint cho TestResults

// ========== INDEXES ==========
// Indexes giúp tối ưu hiệu suất truy vấn

// Indexes for Patient properties
CREATE INDEX patient_age_idx IF NOT EXISTS FOR (p:Patient) ON (p.age);
CREATE INDEX patient_gender_idx IF NOT EXISTS FOR (p:Patient) ON (p.gender);
CREATE INDEX patient_blood_type_idx IF NOT EXISTS FOR (p:Patient) ON (p.blood_type);
CREATE INDEX patient_admission_type_idx IF NOT EXISTS FOR (p:Patient) ON (p.admission_type);
CREATE INDEX patient_date_of_admission_idx IF NOT EXISTS FOR (p:Patient) ON (p.date_of_admission);
CREATE INDEX patient_discharge_date_idx IF NOT EXISTS FOR (p:Patient) ON (p.discharge_date);

// Indexes for TestResults
CREATE INDEX test_outcome_idx IF NOT EXISTS FOR (t:TestResults) ON (t.test_outcome);

// ========== SCHEMA VISUALIZATION ==========
// Chạy lệnh này để kiểm tra schema sau khi tạo
CALL db.schema.visualization();
CALL apoc.meta.schema();

// ========== DATA MODEL DESCRIPTION ==========
// Mô tả cấu trúc dữ liệu của Healthcare GraphRAG

// Node Types và Properties
//
// Patient: {name, age, gender, blood_type, admission_type, date_of_admission, discharge_date}
// Disease: {name}
// Doctor: {name}
// Hospital: {name}
// InsuranceProvider: {name}
// Room: {room_number}
// Medication: {name}
// TestResults: {test_outcome}
// Billing: {amount}

// Relationship Types
//
// Patient Relationships:
// (:Patient)-[:HAS_DISEASE]->(:Disease)
// (:Patient)-[:TREATED_BY]->(:Doctor)
// (:Patient)-[:ADMITTED_TO]->(:Hospital)
// (:Patient)-[:COVERED_BY]->(:InsuranceProvider)
// (:Patient)-[:STAY_IN]->(:Room)
// (:Patient)-[:TAKE_MEDICATION]->(:Medication)
// (:Patient)-[:UNDERGOES]->(:TestResults)
// (:Patient)-[:HAS_BILLING]->(:Billing)
//
// Doctor Relationships:
// (:Doctor)-[:WORKS_AT]->(:Hospital)
// (:Doctor)-[:PRESCRIBES]->(:Medication)
//
// Medication Relationships:
// (:Medication)-[:RELATED_TO_TEST]->(:TestResults)
//
// Hospital Relationships:
// (:Hospital)-[:PARTNERS_WITH]->(:InsuranceProvider)
//
// Billing Relationships:
// (:Billing)-[:COVERED_BY]->(:InsuranceProvider)