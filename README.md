# Healthcare GraphRAG 🏥 

## Chatbot y tế thông minh sử dụng đồ thị tri thức và AI

![Healthcare GraphRAG](https://img.shields.io/badge/Healthcare-GraphRAG-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Neo4j](https://img.shields.io/badge/Database-Neo4j-brightgreen)
![Azure OpenAI](https://img.shields.io/badge/AI-Azure_OpenAI-orange)
![LangChain](https://img.shields.io/badge/Framework-LangChain-yellow)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-purple)

Healthcare GraphRAG là một hệ thống chatbot thông minh kết hợp cơ sở dữ liệu đồ thị Neo4j với các mô hình ngôn ngữ lớn (LLM) từ Azure OpenAI để trả lời các câu hỏi y tế chính xác và có ngữ cảnh. Dự án sử dụng kỹ thuật Retrieval-Augmented Generation (RAG) dựa trên đồ thị tri thức.

## 📋 Mục lục

- [Tính năng nổi bật](#-tính-năng-nổi-bật)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Demo](#-demo)
- [Cài đặt và triển khai](#-cài-đặt-và-triển-khai)
- [Giao diện sử dụng](#-giao-diện-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

## ✨ Tính năng nổi bật

- **Truy vấn thông minh**: Tự động chuyển đổi câu hỏi ngôn ngữ tự nhiên thành truy vấn Cypher chính xác
- **Cơ chế ReAct Agent**: Lựa chọn thông minh giữa RAG và LLM tùy theo loại câu hỏi
- **Đa ngữ**: Hỗ trợ tiếng Việt và tiếng Anh
- **Lưu trữ hội thoại**: Lưu và quản lý các cuộc hội thoại trong cơ sở dữ liệu Neo4j
- **Đa nền tảng**: Giao diện web (Streamlit), API (FastAPI) và CLI
- **Hệ thống bộ nhớ**: Duy trì ngữ cảnh và lịch sử hội thoại
- **Giải thích lý luận**: Hiển thị quá trình suy luận thông qua các truy vấn Cypher
## **System Architecture**
![System Architecture](./Users/quanglekim/Downloads/repo_final/assets/images/graphrag.png)

## 🏗 Kiến trúc hệ thống

Healthcare GraphRAG là một ứng dụng theo mô hình kiến trúc phân lớp với các thành phần chính:

1. **Lớp giao diện người dùng**: 
   - Giao diện web tương tác (Streamlit)
   - API RESTful (FastAPI) 
   - Giao diện dòng lệnh (CLI)

2. **Lớp xử lý**:
   - ReAct Agent đưa ra quyết định sử dụng công cụ nào
   - Trình quản lý bộ nhớ và lịch sử hội thoại
   - Cơ chế theo dõi và phân tích việc sử dụng (LangSmith)

3. **Lớp công cụ**:
   - GraphRAG (Truy xuất dữ liệu từ Neo4j và tăng cường câu trả lời)
   - LLM Tool (Xử lý truy vấn kiến thức chung)

4. **Lớp dữ liệu**:
   - Neo4j Graph Database (dữ liệu y tế và lịch sử hội thoại)
   - Azure OpenAI (Mô hình ngôn ngữ)

## 🎬 Demo

### Giao diện Streamlit
![Streamlit UI Demo](./Users/quanglekim/Downloads/repo_final/assets/images/Screenshot1.png)

### Ví dụ hội thoại
