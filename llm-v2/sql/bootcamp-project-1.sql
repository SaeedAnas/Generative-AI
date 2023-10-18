/*
 * # -------------------------------------------------------------------------------------------------
 * #  Copyright (c) 2023.  SupportVectors AI Lab
 * #  This code is part of the training material, and therefore part of the intellectual property.
 * #  It may not be reused or shared without the explicit, written permission of SupportVectors.
 * #
 * #  Use is limited to the duration and purpose of the training at SupportVectors.
 * #
 * #  Author: Asif Qamar
 * # -------------------------------------------------------------------------------------------------
 *
 */

-- Create the database schema
CREATE DATABASE documents;
USE documents;

-- documents.DOCUMENT definition



-- Create the DOCUMENT table
CREATE TABLE DOCUMENT (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    TEXT LONGTEXT,
    PATH VARCHAR(255),
    LANGUAGE VARCHAR(255),
    UUID VARCHAR(255),
    CHUNKED TINYINT DEFAULT 0,
    DOCTYPE VARCHAR(255)
);

-- Create the CHUNK table
CREATE TABLE CHUNK (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    DOC_ID INT NOT NULL,
    TEXT LONGTEXT,
    VECTORIZED TINYINT DEFAULT 0,
    ANN_INDEXED TINYINT DEFAULT 0,
    ES_INDEXED TINYINT DEFAULT 0,    
    VECTOR TEXT  -- Assuming VECTOR is a text-based representation; adjust type if needed
);

-- Create the METADATA table
CREATE TABLE METADATA (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    DOC_ID INT NOT NULL,
    NAME VARCHAR(255),
    VALUE TEXT
);
