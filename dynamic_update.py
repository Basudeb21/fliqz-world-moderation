#dynamic_update.py
from datetime import datetime
from sqlalchemy import select, update, insert
from database import get_db
from dynamic_table_loader import get_dynamic_table

def dynamic_update(payload: dict, animal_detected=False, das_detected=False, minor_detected=False, personal_info_detected=False, nsfw_detected=False, violence_detected=False, weapon_detected=False, ai_process_status=None):
    """
    Generic UPSERT based on table_name, primary_key, key_value.
    Works for ANY table.
    """
    table_name = payload["table_name"]
    pk_name = payload["primary_key"]
    pk_value = payload["key_value"]

    # Dynamically load table
    table = get_dynamic_table(table_name)

    db = next(get_db())
    now = datetime.now()

    try:
        # 1. Check if row exists
        stmt = select(table).where(table.c[pk_name] == pk_value)
        existing = db.execute(stmt).fetchone()

        # 2. If row exists → UPDATE
        if existing:
            update_data = {
                k: v for k, v in payload.items()
                if k not in ["table_name", "primary_key", "key_value"]
                and k in table.c
            }
            if "updated_at" in table.c:
                update_data["updated_at"] = now

            # Add moderation flags to update_data if columns exist
            if "animal_detected" in table.c:
                update_data["animal_detected"] = 1 if animal_detected else 0

            if "is_das_detected" in table.c:
                update_data["is_das_detected"] = 1 if das_detected else 0

            if "minor_detected" in table.c:
                update_data["minor_detected"] = 1 if minor_detected else 0

            if "is_personal_details_detected" in table.c:
                update_data["is_personal_details_detected"] = 1 if personal_info_detected else 0

            if "nsfw_detected" in table.c:
                update_data["nsfw_detected"] = 1 if nsfw_detected else 0

            if "violance_detected" in table.c:
                update_data["violance_detected"] = 1 if violence_detected else 0

            if "is_weapon_detected" in table.c:
                update_data["is_weapon_detected"] = 1 if weapon_detected else 0

            # -----------------------------
            # AI PROCESS STATUS
            # -----------------------------
            if ai_process_status is not None and "ai_process_status" in table.c:
                update_data["ai_process_status"] = ai_process_status

            # -----------------------------
            # BLOCKING LOGIC
            # -----------------------------
            is_blocked = 0

            if (
                minor_detected or
                personal_info_detected or
                animal_detected or
                violence_detected or
                das_detected or
                weapon_detected
            ):
                is_blocked = 1

            if "is_blocked" in table.c:
                update_data["is_blocked"] = is_blocked

            stmt = (
                update(table)
                .where(table.c[pk_name] == pk_value)
                .values(update_data)
            )

            db.execute(stmt)
            db.commit()
            return True, "updated"

        else:
            return False, "row_not_found"

    except Exception as e:
        db.rollback()
        return False, str(e)


def insert_detection_timestamps(payload: dict, records: list, table_name: str = "moderation_timestamps"):
    """
    Insert detection timestamp records into a separate table.

    records: list of dicts with keys: post_id, label, start_time, end_time, duration
    This function attempts to insert rows and fails silently (returns False,msg)
    if the target table does not exist or insertion fails.
    """
    if not records:
        return False, "no_records"

    try:
        table = get_dynamic_table(table_name)
    except Exception as e:
        return False, f"table_load_failed: {e}"

    db = next(get_db())
    try:
        # attempt to resolve attachment_id for records when possible
        attachments_table = None
        for rec in records:
            # ensure post_id exists in record, fallback to payload.data.post_id
            post_id = rec.get("post_id") or payload.get("data", {}).get("post_id") or payload.get("post_id")

            insert_row = {
                k: v for k, v in rec.items()
                if k in table.c
            }

            # include post_id if column exists
            if "post_id" in table.c and "post_id" not in insert_row:
                insert_row["post_id"] = post_id

            # If moderation_timestamps has attachment_id column, try to find attachment id by post_id
            if "attachment_id" in table.c and "attachment_id" not in insert_row:
                try:
                    if attachments_table is None:
                        attachments_table = get_dynamic_table("attachments")

                    if post_id is not None:
                        stmt = select(attachments_table).where(attachments_table.c.post_id == post_id)
                        found = db.execute(stmt).fetchone()
                        if found and "id" in attachments_table.c:
                            insert_row["attachment_id"] = found[attachments_table.c.id]
                except Exception:
                    # best-effort — if attachments table missing or query fails, skip
                    pass

            # deduplicate: skip if an identical row already exists
            try:
                dup_clause = []

                # Always deduplicate on post_id + label — these two together are the
                # minimum identity of a detection event. All other fields are optional.
                if "post_id" in table.c:
                    dup_clause.append(table.c.post_id == insert_row.get("post_id"))
                if "label" in table.c:
                    dup_clause.append(table.c.label == insert_row.get("label"))

                # Only add time/type fields to the clause if they are actually non-None.
                # If they are None and included, the WHERE becomes "col = NULL" which
                # never matches in SQL — the dedup check silently fails every time.
                if insert_row.get("start_time") is not None and "start_time" in table.c:
                    dup_clause.append(table.c.start_time == insert_row["start_time"])
                if insert_row.get("end_time") is not None and "end_time" in table.c:
                    dup_clause.append(table.c.end_time == insert_row["end_time"])
                if insert_row.get("duration") is not None and "duration" in table.c:
                    dup_clause.append(table.c.duration == insert_row["duration"])
                if insert_row.get("detector_type") is not None and "detector_type" in table.c:
                    dup_clause.append(table.c.detector_type == insert_row["detector_type"])

                duplicate = False
                if dup_clause:
                    sel = select(table).where(*dup_clause)
                    found_dup = db.execute(sel).fetchone()
                    if found_dup:
                        duplicate = True

                if duplicate:
                    # skip inserting duplicate
                    continue

            except Exception:
                # if dedupe check fails for any reason, fall back to inserting
                pass

            # attempt insert
            stmt = insert(table).values(**insert_row)
            db.execute(stmt)

        db.commit()
        return True, "inserted"

    except Exception as e:
        db.rollback()
        return False, str(e)