"""create college_history_chunks table

Revision ID: d7e2b9f4a1c8
Revises: c2d8e4a1f6b7
Create Date: 2026-05-21

Stores chunks of source college-history text for RAG retrieval. On an FAQ miss
the top-K most similar chunks are injected into the LLM prompt as grounding.

The HNSW index keeps cosine search O(log n) as the knowledge base grows into
thousands of chunks (mirrors the FAQ embedding index).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = 'd7e2b9f4a1c8'
down_revision: Union[str, Sequence[str], None] = 'c2d8e4a1f6b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'college_history_chunks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(384), nullable=True),
        sa.Column('character_id', sa.String(length=10), nullable=True),
        sa.Column('source_doc', sa.Text(), nullable=True),
        sa.Column('chunk_index', sa.Integer(), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=False, server_default='en'),
        sa.Column('tag', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_college_history_chunks_character_id'),
        'college_history_chunks',
        ['character_id'],
        unique=False,
    )
    op.execute("""
        CREATE INDEX IF NOT EXISTS college_history_embedding_hnsw_idx
        ON college_history_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS college_history_embedding_hnsw_idx")
    op.drop_index(op.f('ix_college_history_chunks_character_id'), table_name='college_history_chunks')
    op.drop_table('college_history_chunks')
