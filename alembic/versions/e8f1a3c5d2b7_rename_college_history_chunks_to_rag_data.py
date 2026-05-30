"""rename college_history_chunks to RAG_data

Revision ID: e8f1a3c5d2b7
Revises: d7e2b9f4a1c8
Create Date: 2026-05-31

Renames the college_history_chunks RAG store to RAG_data. The name is
mixed-case, so PostgreSQL stores it quoted (case-sensitive) — every raw-SQL
reference must wrap it in double quotes. Indexes are renamed alongside the
table to match SQLAlchemy's naming convention.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'e8f1a3c5d2b7'
down_revision: Union[str, Sequence[str], None] = 'd7e2b9f4a1c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.rename_table('college_history_chunks', 'RAG_data')
    op.execute('ALTER INDEX ix_college_history_chunks_character_id RENAME TO "ix_RAG_data_character_id"')
    op.execute('ALTER INDEX college_history_embedding_hnsw_idx RENAME TO rag_data_embedding_hnsw_idx')


def downgrade() -> None:
    op.execute('ALTER INDEX rag_data_embedding_hnsw_idx RENAME TO college_history_embedding_hnsw_idx')
    op.execute('ALTER INDEX "ix_RAG_data_character_id" RENAME TO ix_college_history_chunks_character_id')
    op.rename_table('RAG_data', 'college_history_chunks')
