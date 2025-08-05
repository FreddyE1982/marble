from kuzu_interface import KuzuGraphDatabase


def setup_db(tmp_path):
    db_path = tmp_path / "kuzu_db"
    db = KuzuGraphDatabase(str(db_path))
    db.create_node_table("Person", {"name": "STRING", "age": "INT64"}, "name")
    db.create_node_table("City", {"name": "STRING"}, "name")
    db.create_relationship_table("LIVES_IN", "Person", "City", {"since": "INT64"})
    return db, db_path


def test_create_and_query(tmp_path):
    db, _ = setup_db(tmp_path)
    db.add_node("Person", {"name": "Alice", "age": 30})
    db.add_node("City", {"name": "Wonderland"})
    db.add_relationship(
        "Person",
        "name",
        "Alice",
        "LIVES_IN",
        "City",
        "name",
        "Wonderland",
        {"since": 2024},
    )
    rows = db.execute(
        "MATCH (p:Person)-[r:LIVES_IN]->(c:City) "
        "RETURN p.name AS pname, p.age AS age, c.name AS city, r.since AS since;"
    )
    assert rows == [{"pname": "Alice", "age": 30, "city": "Wonderland", "since": 2024}]


def test_update_and_delete(tmp_path):
    db, _ = setup_db(tmp_path)
    db.add_node("Person", {"name": "Bob", "age": 25})
    db.update_node("Person", "name", "Bob", {"age": 26})
    rows = db.execute(
        "MATCH (p:Person {name:$name}) RETURN p.age AS age", {"name": "Bob"}
    )
    assert rows[0]["age"] == 26
    db.add_node("City", {"name": "NY"})
    db.add_relationship("Person", "name", "Bob", "LIVES_IN", "City", "name", "NY")
    db.delete_relationship("Person", "name", "Bob", "LIVES_IN", "City", "name", "NY")
    rows = db.execute(
        "MATCH (:Person {name:$name})-[:LIVES_IN]->() RETURN count(*) AS cnt",
        {"name": "Bob"},
    )
    assert rows[0]["cnt"] == 0
    db.delete_node("Person", "name", "Bob")
    rows = db.execute(
        "MATCH (p:Person {name:$name}) RETURN count(*) AS cnt", {"name": "Bob"}
    )
    assert rows[0]["cnt"] == 0


def test_persistence(tmp_path):
    db, path = setup_db(tmp_path)
    db.add_node("Person", {"name": "Eve", "age": 22})
    db.close()
    db2 = KuzuGraphDatabase(str(path))
    rows = db2.execute("MATCH (p:Person) RETURN p.name AS name")
    assert {"name": "Eve"} in rows
    db2.close()
