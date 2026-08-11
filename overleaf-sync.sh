#!/usr/bin/env bash
#
# Sincroniza a pasta da dissertação (dissertacao/overleaf-dissertacao) com o
# projeto Overleaf, usando git subtree. O token de acesso ao Overleaf fica
# guardado no keychain do macOS (não neste arquivo, não no repositório).
#
#   ./overleaf-sync.sh pull       # traz para o repo as edições feitas no Overleaf
#   ./overleaf-sync.sh push       # envia para o Overleaf os commits feitos no repo
#   ./overleaf-sync.sh status     # mostra as diferenças, sem alterar nada
#   ./overleaf-sync.sh bootstrap  # (uma única vez) sobrescreve o Overleaf com o repo
#
# Fluxo recomendado: sempre 'pull' ANTES de editar e 'push' DEPOIS de commitar.
#
set -euo pipefail

PREFIX="dissertacao/overleaf-dissertacao"
REMOTE="overleaf"
BRANCH="main"

cd "$(git rev-parse --show-toplevel)"

require_clean() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERRO: há mudanças não commitadas. Faça commit antes de sincronizar." >&2
    git status --short
    exit 1
  fi
}

case "${1:-}" in
  pull)
    require_clean
    git fetch "$REMOTE" "$BRANCH"
    git subtree pull --prefix="$PREFIX" "$REMOTE" "$BRANCH" -m "sync: pull do Overleaf"
    echo "OK. Lembre de 'git push origin main' para levar as mudanças ao GitHub."
    ;;
  push)
    require_clean
    git subtree push --prefix="$PREFIX" "$REMOTE" "$BRANCH"
    echo "OK. Overleaf atualizado com o estado do repositório."
    ;;
  status)
    git fetch "$REMOTE" "$BRANCH" >/dev/null 2>&1 || true
    echo "Diferenças entre o Overleaf (remoto) e o repo local, na pasta $PREFIX:"
    git diff --stat "$REMOTE/$BRANCH" "HEAD:$PREFIX" 2>/dev/null \
      || echo "(não foi possível comparar; rode 'bootstrap' se ainda não sincronizou)"
    ;;
  bootstrap)
    # Vincula o repo ao Overleaf SEM force-push (o Overleaf proíbe force).
    # Estratégia: monta um histórico que descende do head atual do Overleaf,
    # mas com o CONTEÚDO do repositório (merge -s ours), e faz push normal.
    require_clean
    echo "Bootstrap: vincula o repositório ao Overleaf (push normal, sem force)."
    echo ">> IMPORTANTE: NÃO edite o projeto no Overleaf enquanto isto roda."
    echo "   (Cada edição no editor vira um commit e faz o push falhar.)"
    read -r -p "Digite 'sim' para continuar: " ans
    [ "$ans" = "sim" ] || { echo "Cancelado."; exit 1; }
    git fetch "$REMOTE" "$BRANCH"
    git branch -D _ovl_tmp >/dev/null 2>&1 || true
    git subtree split --prefix="$PREFIX" -b _ovl_tmp
    git checkout _ovl_tmp
    git merge --allow-unrelated-histories -s ours "$REMOTE/$BRANCH" \
      -m "join: histórico do Overleaf (conteúdo reconciliado do repositório)"
    if ! git push "$REMOTE" "_ovl_tmp:$BRANCH"; then
      git checkout main
      git branch -D _ovl_tmp >/dev/null 2>&1 || true
      echo "ERRO: push rejeitado (o Overleaf mudou durante o processo)."
      echo "Feche o editor do Overleaf, espere alguns segundos e rode o bootstrap de novo."
      exit 1
    fi
    git checkout main
    git subtree pull --prefix="$PREFIX" "$REMOTE" "$BRANCH" -m "sync: join inicial com o Overleaf"
    git branch -D _ovl_tmp >/dev/null 2>&1 || true
    echo "OK. Repositório vinculado ao Overleaf."
    echo "Rode 'git push origin main' para levar o join ao GitHub."
    ;;
  *)
    echo "uso: $0 {pull|push|status|bootstrap}" >&2
    exit 1
    ;;
esac
