"""BlobTools — Agno Toolkit for brain access to BlobStore."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agno.tools import Toolkit

from enton.core.blob_store import BlobType

if TYPE_CHECKING:
    from enton.core.blob_store import BlobStore


class BlobTools(Toolkit):
    def __init__(self, blob_store: BlobStore) -> None:
        super().__init__(name="blob_tools")
        self._store = blob_store
        self.register(self.search_blobs)
        self.register(self.recent_blobs)
        self.register(self.blob_stats)

    async def search_blobs(self, query: str, blob_type: str = "") -> str:
        """Busca nos arquivos binarios armazenados no HD externo do Enton.

        Args:
            query: O que buscar (ex: "foto do Gabriel", "audio da conversa").
            blob_type: Tipo opcional: images, audio, video, faces, snapshots.
        """
        bt = BlobType(blob_type) if blob_type else None
        results = await self._store.search(query, blob_type=bt, n=5)
        if not results:
            return "Nenhum blob encontrado."
        return "\n".join(
            f"- [{m.blob_type.value}] {m.blob_id} ({m.size_bytes}B) tags={m.tags}"
            for m in results
        )

    async def recent_blobs(self, n: int = 5, blob_type: str = "") -> str:
        """Lista os blobs mais recentes armazenados no HD externo.

        Args:
            n: Quantidade de resultados (max 20).
            blob_type: Filtro por tipo: images, audio, video, faces, snapshots.
        """
        bt = BlobType(blob_type) if blob_type else None
        results = await self._store.recent(blob_type=bt, n=min(n, 20))
        if not results:
            return "Nenhum blob recente."
        return "\n".join(
            f"- [{m.blob_type.value}] {m.blob_id} ({m.size_bytes}B)"
            for m in results
        )

    async def blob_stats(self) -> str:
        """Mostra estatisticas do armazenamento de blobs no HD externo."""
        stats = await self._store.stats()
        return json.dumps(stats, indent=2, ensure_ascii=False)
