"""Kioku Lite CLI — zero Docker, zero cloud LLM.

Commands:
  save            Save a memory. Returns content_hash for kg-index.
  search          Tri-hybrid search (BM25 + Vector + Graph).
  kg-index        Agent-provided entity/relationship indexing for a saved memory.
  kg-alias        Register SAME_AS aliases for a canonical entity.
  kg-invalidate   Mark a knowledge graph edge as no longer valid.
  dedup-scan      Scan entities for near-duplicates; optionally auto-merge.
  merge           Manually merge two entities (source -> target).
  recall          Recall everything related to an entity.
  connect         Explain connection between two entities.
  entities        List top entities in the knowledge graph.
  timeline        Chronological memory list.
  export-graph    Export knowledge graph as interactive HTML or JSON.
  setup           First-time setup (download embedding model, create config).
  init            Generate CLAUDE.md + SKILL.md for Claude Code / Cursor.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    raise ImportError("Install with: pip install kioku-agent-kit-lite[cli]")

app = typer.Typer(
    name="kioku-lite",
    help="Personal memory agent — zero Docker, zero cloud LLM.",
    no_args_is_help=True,
)

_svc = None


def _get_svc():
    global _svc
    if _svc is None:
        from kioku_lite.service import KiokuLiteService
        _svc = KiokuLiteService()
    return _svc


def _out(data: dict | str) -> None:
    if isinstance(data, str):
        typer.echo(data)
    else:
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


# ── save ───────────────────────────────────────────────────────────────────────

@app.command()
def save(
    text: str = typer.Argument(..., help="Memory text to save."),
    mood: Optional[str] = typer.Option(None, "--mood", "-m"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags."),
    event_time: Optional[str] = typer.Option(None, "--event-time", "-e", help="When it happened (YYYY-MM-DD)."),
) -> None:
    """Save a memory. Prints content_hash — use it with kg-index to add KG entries."""
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    result = _get_svc().save_memory(text, mood=mood, tags=tag_list, event_time=event_time)
    _out(result)


# ── kg-index ───────────────────────────────────────────────────────────────────

@app.command(name="kg-index")
def kg_index(
    content_hash: str = typer.Argument(..., help="content_hash returned by `save`."),
    entities: Optional[str] = typer.Option(
        None, "--entities", "-e",
        help='JSON array of entities. Example: \'[{"name":"Phúc","type":"PERSON"}]\''
    ),
    relationships: Optional[str] = typer.Option(
        None, "--relationships", "-r",
        help='JSON array of relationships. Example: \'[{"source":"Phúc","target":"TBV","rel_type":"WORKS_AT","weight":0.8,"evidence":"..."}]\''
    ),
    event_time: Optional[str] = typer.Option(None, "--event-time", help="YYYY-MM-DD when the event happened."),
) -> None:
    """Index agent-extracted entities and relationships for a saved memory.

    The agent (Claude Code / OpenClaw) calls this AFTER `save`, providing
    entities and relationships it already extracted from conversation context.
    Kioku Lite does NOT call any LLM — it simply stores what the agent provides.

    Example workflow (agent SKILL.md):

      1. hash = `kioku-lite save "Hôm nay gặp Hùng ở TBV..." --mood happy`
      2. Agent extracts: entities=[Hùng/PERSON, TBV/PLACE], rel=[Hùng→TBV WORKS_AT]
      3. `kioku-lite kg-index <hash> --entities '[...]' --relationships '[...]'`
    """
    from kioku_lite.service import EntityInput, RelationshipInput

    entity_list: list[EntityInput] = []
    if entities:
        try:
            raw = json.loads(entities)
            entity_list = [
                EntityInput(name=e["name"], type=e.get("type", "TOPIC"),
                            confidence=float(e.get("confidence", 1.0)))
                for e in raw
            ]
        except (json.JSONDecodeError, KeyError) as err:
            typer.echo(f"Error parsing --entities JSON: {err}", err=True)
            raise typer.Exit(1)

    rel_list: list[RelationshipInput] = []
    if relationships:
        try:
            raw = json.loads(relationships)
            rel_list = [
                RelationshipInput(
                    source=r["source"], target=r["target"],
                    rel_type=r.get("rel_type", "TOPICAL"),
                    weight=r.get("weight", 0.5),
                    evidence=r.get("evidence", ""),
                )
                for r in raw
            ]
        except (json.JSONDecodeError, KeyError) as err:
            typer.echo(f"Error parsing --relationships JSON: {err}", err=True)
            raise typer.Exit(1)

    result = _get_svc().kg_index(content_hash, entity_list, rel_list, event_time=event_time)
    _out(result)


# ── kg-alias ───────────────────────────────────────────────────────────────────

@app.command(name="kg-alias")
def kg_alias(
    canonical: str = typer.Argument(..., help="The canonical entity name."),
    aliases: str = typer.Option(
        ..., "--aliases", "-a",
        help='JSON array of alias names. Example: \'["phuc-nt","anh","Phúc"]\''
    ),
) -> None:
    """Register SAME_AS aliases for a canonical entity.

    Example:
      kioku-lite kg-alias "Nguyễn Trọng Phúc" --aliases '["phuc-nt","Phúc","anh","tôi"]'
    """
    try:
        alias_list = json.loads(aliases)
    except json.JSONDecodeError as err:
        typer.echo(f"Error parsing --aliases JSON: {err}", err=True)
        raise typer.Exit(1)
    _out(_get_svc().kg_alias(canonical, alias_list))


# ── dedup-scan ────────────────────────────────────────────────────────────────

@app.command(name="dedup-scan")
def dedup_scan(
    auto: bool = typer.Option(False, "--auto", help="Automatically merge qualifying pairs."),
) -> None:
    """Scan all entities for near-duplicates and surface merge candidates.

    With --auto: auto-merges pairs that meet both vector and name similarity
    thresholds (vec >= 0.98 AND name >= 0.85).

    \b
    Examples:
      kioku-lite dedup-scan
      kioku-lite dedup-scan --auto
    """
    _out(_get_svc().dedup_scan(auto_merge=auto))


# ── merge ─────────────────────────────────────────────────────────────────────

@app.command()
def merge(
    source: str = typer.Argument(..., help="Entity name to merge FROM (will be deleted)."),
    target: str = typer.Argument(..., help="Entity name to merge INTO (will be kept)."),
) -> None:
    """Manually merge two entities: source is absorbed into target.

    All edges are re-pointed, source is registered as an alias, and
    the source node is removed.

    \b
    Example:
      kioku-lite merge "Phuc" "Phúc"
    """
    _out(_get_svc().merge_entities(source, target))


# ── kg-invalidate ─────────────────────────────────────────────────────────────

@app.command(name="kg-invalidate")
def kg_invalidate(
    source: str = typer.Option(..., "--source", "-s", help="Source entity."),
    target: str = typer.Option(..., "--target", "-t", help="Target entity."),
    rel_type: Optional[str] = typer.Option(None, "--rel-type", "-r", help="Relationship type (optional)."),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="When fact became invalid (YYYY-MM-DD). Defaults to today."),
    reason: Optional[str] = typer.Option("", "--reason", help="Why this fact is no longer valid."),
) -> None:
    """Mark a knowledge graph edge as no longer valid (superseded).

    \b
    Example — Phuc changed jobs:
      kioku-lite kg-invalidate --source Phuc --target LINE --rel-type WORKS_AT --date 2026-03-31
    """
    result = _get_svc().kg_invalidate(
        source=source, target=target, rel_type=rel_type,
        valid_until=date, reason=reason or "",
    )
    _out(result)


# ── search ─────────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="What to search for."),
    limit: int = typer.Option(10, "--limit", "-l"),
    date_from: Optional[str] = typer.Option(None, "--from", help="Start date YYYY-MM-DD."),
    date_to: Optional[str] = typer.Option(None, "--to", help="End date YYYY-MM-DD."),
    entities: Optional[str] = typer.Option(None, "--entities", "-e", help="Comma-separated entity names for KG seeding."),
    include_historical: bool = typer.Option(False, "--include-historical", help="Include superseded facts."),
) -> None:
    """Tri-hybrid search: BM25 + Vector (FastEmbed) + Knowledge Graph → RRF rerank."""
    entity_list = [e.strip() for e in entities.split(",")] if entities else None
    _out(_get_svc().search_memories(
        query, limit=limit, date_from=date_from, date_to=date_to,
        entities=entity_list, include_historical=include_historical,
    ))


# ── recall ─────────────────────────────────────────────────────────────────────

@app.command()
def recall(
    entity: str = typer.Argument(..., help="Entity name to recall memories for."),
    hops: int = typer.Option(2, "--hops", help="Graph traversal depth."),
    limit: int = typer.Option(10, "--limit", "-l"),
    include_historical: bool = typer.Option(False, "--include-historical", help="Include superseded facts."),
) -> None:
    """Recall all memories related to an entity via knowledge graph traversal."""
    _out(_get_svc().recall_entity(entity, max_hops=hops, limit=limit, include_historical=include_historical))


# ── connect ────────────────────────────────────────────────────────────────────

@app.command()
def connect(
    entity_a: str = typer.Argument(...),
    entity_b: str = typer.Argument(...),
    include_historical: bool = typer.Option(False, "--include-historical", help="Include superseded facts."),
) -> None:
    """Explain how two entities are connected in the knowledge graph."""
    _out(_get_svc().explain_connection(entity_a, entity_b, include_historical=include_historical))


# ── entities ───────────────────────────────────────────────────────────────────

@app.command()
def entities(
    limit: int = typer.Option(50, "--limit", "-l"),
) -> None:
    """List top canonical entities from the knowledge graph."""
    _out(_get_svc().list_entities(limit=limit))


# ── timeline ───────────────────────────────────────────────────────────────────

@app.command()
def timeline(
    start_date: Optional[str] = typer.Option(None, "--from"),
    end_date: Optional[str] = typer.Option(None, "--to"),
    limit: int = typer.Option(50, "--limit", "-l"),
    sort_by: str = typer.Option("processing_time", "--sort-by", "-s", help="'processing_time' or 'event_time'."),
) -> None:
    """Chronological memory list."""
    _out(_get_svc().get_timeline(start_date=start_date, end_date=end_date, limit=limit, sort_by=sort_by))


# ── users ──────────────────────────────────────────────────────────────────────

@app.command()
def users(
    create: Optional[str] = typer.Option(None, "--create", "-c", help="Create a new profile."),
    use: Optional[str] = typer.Option(None, "--use", "-u", help="Set active profile for this session."),
) -> None:
    """List all user profiles, create a new one, or switch active profile.

    Run at the start of each session to pick which memory profile to use.
    After --use, all subsequent kioku-lite commands use that profile automatically.

    \b
    # List profiles:
    kioku-lite users

    # Switch to a profile (no prefix needed after this):
    kioku-lite users --use work

    # Create a new profile:
    kioku-lite users --create work
    """
    base_dir = Path.home() / ".kioku-lite" / "users"
    active_file = Path.home() / ".kioku-lite" / ".active_user"

    # ── --use: set active session profile ────────────────────────────────────
    if use:
        # Profile must exist
        if not (base_dir / use).exists() and use != "personal":
            typer.echo(f"⚠️  Profile '{use}' not found. Create it first: kioku-lite users --create {use}", err=True)
            raise typer.Exit(1)
        # Ensure dir exists for personal
        (base_dir / use / "data").mkdir(parents=True, exist_ok=True)
        (base_dir / use / "memory").mkdir(parents=True, exist_ok=True)
        # Write active profile file
        active_file.parent.mkdir(parents=True, exist_ok=True)
        active_file.write_text(use)
        _out({"status": "active", "user_id": use, "note": "All subsequent kioku-lite commands will use this profile."})
        return

    # ── --create: make new profile ────────────────────────────────────────────
    if create:
        if not create.replace("-", "").replace("_", "").isalnum():
            typer.echo("⚠️  Profile ID can only contain letters, numbers, hyphens and underscores.", err=True)
            raise typer.Exit(1)
        profile_dir = base_dir / create / "data"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / create / "memory").mkdir(parents=True, exist_ok=True)
        _out({"status": "created", "user_id": create, "path": str(base_dir / create)})
        typer.echo(f"\nActivate it now: kioku-lite users --use {create}")
        return

    # ── list: show all profiles ───────────────────────────────────────────────
    # Ensure default profile "personal" always exists
    (base_dir / "personal" / "data").mkdir(parents=True, exist_ok=True)
    (base_dir / "personal" / "memory").mkdir(parents=True, exist_ok=True)

    # Read current active profile
    current_active = active_file.read_text().strip() if active_file.exists() else "personal"

    profiles = []
    if base_dir.exists():
        for p in sorted(base_dir.iterdir()):
            if p.is_dir():
                db = p / "data" / "kioku.db"
                profiles.append({
                    "user_id": p.name,
                    "active": p.name == current_active,
                    "has_data": db.exists(),
                    "db_size_kb": round(db.stat().st_size / 1024, 1) if db.exists() else 0,
                })

    _out({
        "profiles": profiles,
        "active_profile": current_active,
        "hint": "Run 'kioku-lite users --use <user_id>' to switch profiles",
    })

# ── setup ──────────────────────────────────────────────────────────────────────

@app.command()
def setup() -> None:
    """Pre-download the embedding model (~1.1GB, first-time only).

    Optional — kioku-lite works without running this. The model downloads
    automatically on first use. Run this to eagerly pre-download before
    going offline, or to verify the local install.

    \b
    Profile management: kioku-lite users
    Agent integration:  kioku-lite init --global
    """
    embed_model = "intfloat/multilingual-e5-large"

    typer.echo("")
    typer.echo("╔══════════════════════════════════════╗")
    typer.echo("║   Kioku Lite — Pre-download Model    ║")
    typer.echo("╚══════════════════════════════════════╝")
    typer.echo(f"\nModel  : {embed_model}")
    typer.echo("Target : ~/.cache/fastembed/")
    typer.echo("Size   : ~1.1GB (once only)\n")

    try:
        from kioku_lite.pipeline.embedder import FastEmbedder
        embedder = FastEmbedder(model_name=embed_model)
        embedder.embed("warmup")
        typer.echo("\n✅ Model ready.")
    except Exception as e:
        typer.echo(f"\n⚠️  Download failed: {e}")
        typer.echo("   Run again when online.")
        typer.echo("   Or: KIOKU_LITE_EMBED_PROVIDER=fake kioku-lite save '...'  (BM25+Graph only, no vectors)")

    typer.echo("\nNext steps:")
    typer.echo("  kioku-lite users --use personal   # pick active profile")
    typer.echo("  kioku-lite init --global           # inject SKILL.md for Claude Code")
    typer.echo('  kioku-lite save "Your first memory"')
    typer.echo("")





# ── init ───────────────────────────────────────────────────────────────────────

@app.command()
def init(
    global_: bool = typer.Option(
        False, "--global", "-g",
        help="Install globally into ~/.claude/ — works in ALL projects without re-running init.",
    ),
) -> None:
    """Generate CLAUDE.md + SKILL.md for Claude Code / Cursor agent integration.

    By default: writes to the current project directory.

    With --global: installs SKILL.md into ~/.claude/skills/kioku-lite/SKILL.md
    so Claude Code picks it up in EVERY project automatically. Recommended.

    \b
    # One-time global setup (recommended):
    kioku-lite init --global

    # Per-project setup (if you want project-specific):
    kioku-lite init
    """
    RESOURCES = Path(__file__).parent / "resources"

    claude_src = RESOURCES / "CLAUDE.agent.md"
    skill_src = RESOURCES / "SKILL.md"

    if not claude_src.exists() or not skill_src.exists():
        typer.echo("⚠️  Resource files not found. Reinstall: pip install 'kioku-lite[cli]'", err=True)
        raise typer.Exit(1)

    if global_:
        # Write to ~/.claude/ — Claude Code reads this from any project
        skill_dir = Path.home() / ".claude" / "skills" / "kioku-lite"
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_dst = skill_dir / "SKILL.md"
        skill_dst.write_text(skill_src.read_text(encoding="utf-8"))

        typer.echo("")
        typer.echo("✅ Step 1 complete — Global SKILL.md installed:")
        typer.echo(f"   {skill_dst}")
        typer.echo("")
        typer.echo("Your AI agent now knows HOW to use kioku-lite in every project.")
        typer.echo("")
        typer.echo("── Step 2: Activate per workspace ────────────────────────────────────")
        typer.echo("AGENTS.md cannot be global — it's a per-workspace file that ACTIVATES")
        typer.echo("kioku-lite for a specific directory. Run one of these in your workspace:")
        typer.echo("")
        typer.echo("  # Standard (no persona):")
        typer.echo("  cd ~/your-workspace && kioku-lite init")
        typer.echo("")
        typer.echo("  # With a built-in persona (companion / mentor):")
        typer.echo("  cd ~/your-workspace && kioku-lite install-profile companion")
        typer.echo("  cd ~/your-workspace && kioku-lite install-profile mentor")
        typer.echo("")
    else:
        # Write to current project directory — use open standard (AGENTS.md + .agents/skills)
        agents_dst = Path.cwd() / "AGENTS.md"
        skill_dir = Path.cwd() / ".agents" / "skills" / "kioku-lite"
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_dst = skill_dir / "SKILL.md"

        agents_dst.write_text(claude_src.read_text(encoding="utf-8"))
        skill_dst.write_text(skill_src.read_text(encoding="utf-8"))

        typer.echo("")
        typer.echo("✅ Project install:")
        typer.echo(f"   {agents_dst}")
        typer.echo(f"   {skill_dst}")
        typer.echo("")
        typer.echo("Your AI agent will use kioku-lite in THIS project.")
        typer.echo("(Works with Claude Code, Cursor, Windsurf, and any AGENTS.md-aware agent.)")
        typer.echo("")
        typer.echo("Tip: Run `kioku-lite init --global` once to enable in ALL projects.")
        typer.echo("")




# ── install-profile ────────────────────────────────────────────────────────────

@app.command()
def install_profile(
    profile_name: str = typer.Argument(..., help="The profile name to install (e.g. companion, mentor)."),
) -> None:
    """Install an Agent Skill profile (e.g., companion, mentor).
    
    This copies the predefined SKILL.md for the requested profile into 
    ~/.agents/skills/kioku-<profile_name>/SKILL.md, and creates an AGENTS.md 
    in the current directory to activate it.
    """
    RESOURCES = Path(__file__).parent / "resources" / "profiles"
    profile_dir = RESOURCES / profile_name

    if not profile_dir.exists() or not profile_dir.is_dir():
        typer.echo(f"⚠️  Profile '{profile_name}' not found. Available profiles:", err=True)
        if RESOURCES.exists():
            for p in RESOURCES.iterdir():
                if p.is_dir():
                    typer.echo(f"  - {p.name}")
        raise typer.Exit(1)

    skill_src = profile_dir / "SKILL.md"
    agents_src = profile_dir / "AGENTS.md"

    if not skill_src.exists() or not agents_src.exists():
        typer.echo(f"⚠️  Profile '{profile_name}' is incomplete or corrupted.", err=True)
        raise typer.Exit(1)

    # 1. Install SKILL.md into global Agent Skills directory
    dest_skill_dir = Path.home() / ".agents" / "skills" / f"kioku-{profile_name}"
    dest_skill_dir.mkdir(parents=True, exist_ok=True)
    dest_skill_file = dest_skill_dir / "SKILL.md"
    dest_skill_file.write_text(skill_src.read_text(encoding="utf-8"))

    # 2. Install AGENTS.md into current directory
    dest_agents_file = Path.cwd() / "AGENTS.md"
    # If AGENTS.md already exists, append to it instead of overwriting, though it's 
    # generally better to just warn. We'll warn and write a new one if it doesn't exist,
    # or append if the user accepts (simplification: just write if missing, append if present)
    if not dest_agents_file.exists():
        dest_agents_file.write_text(agents_src.read_text(encoding="utf-8"))
        agent_msg = f"Created {dest_agents_file}"
    else:
        # Append separated by a newline
        existing = dest_agents_file.read_text(encoding="utf-8")
        append_text = "\n\n" + agents_src.read_text(encoding="utf-8")
        dest_agents_file.write_text(existing + append_text)
        agent_msg = f"Appended constraints to existing {dest_agents_file}"

    typer.echo("")
    typer.echo(f"✅ Profile '{profile_name}' installed successfully.")
    typer.echo(f"   [{dest_skill_file}] - Agent Skill (Global)")
    typer.echo(f"   [{agent_msg}] - Workspace Context")
    typer.echo("")
    typer.echo(f"Make sure you select the right profile with: kioku-lite users --use {profile_name}")
    typer.echo("")


@app.command(name="export-graph")
def export_graph(
    output: Optional[str] = typer.Argument(
        None, help="Output file path. Defaults to graph.html or graph.json."
    ),
    format_: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Export format: html (interactive, requires pyvis) or json (D3 node-link).",
    ),
) -> None:
    """Export the knowledge graph as an interactive file.

    \b
    Formats:
      html  Standalone interactive HTML (vis-network). Open in any browser.
            Requires: pip install pyvis  (or pip install "kioku-lite[export]")
      json  D3 node-link JSON. Import into any graph tool or web app.

    \b
    Examples:
      kioku-lite export-graph                        # → graph.html
      kioku-lite export-graph mygraph.html           # → mygraph.html
      kioku-lite export-graph --format json          # → graph.json
      kioku-lite export-graph report.json -f json    # → report.json
    """
    from kioku_lite.export_graph import export_html, export_json

    fmt = format_.lower().strip()
    if fmt not in ("html", "json"):
        typer.echo(f"⚠️  Unknown format '{fmt}'. Choose 'html' or 'json'.", err=True)
        raise typer.Exit(1)

    # Resolve default output path
    if output is None:
        output = f"graph.{fmt}"

    svc = _get_svc()
    graph_data = svc.get_graph_data()

    node_count = len(graph_data.get("nodes", []))
    edge_count = len(graph_data.get("links", []))

    if node_count == 0:
        typer.echo(
            "⚠️  No entities found in the knowledge graph.\n"
            "   Save some memories and run 'kioku-lite kg-index' first.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Exporting {node_count} nodes, {edge_count} edges → {fmt.upper()} …")

    try:
        if fmt == "html":
            out = export_html(graph_data, output)
            typer.echo(f"✅ Graph exported: {out}")
            typer.echo("   Open in your browser to explore interactively.")
        else:
            out = export_json(graph_data, output)
            typer.echo(f"✅ Graph exported: {out}")
            typer.echo("   Load into any D3-compatible tool or web app.")
    except ImportError as exc:
        typer.echo(f"⚠️  {exc}", err=True)
        raise typer.Exit(1)
    except ValueError as exc:
        typer.echo(f"⚠️  {exc}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

