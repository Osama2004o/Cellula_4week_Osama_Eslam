import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "app_data.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates a connection to the SQLite database with Row support."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = DB_PATH, force: bool = False) -> None:
    """
    Initializes the SQLite database with realistic seed tables and data
    representing an enterprise e-commerce and company management system.
    """
    if os.path.exists(db_path) and not force:
        # Check if tables exist
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='products'")
            if cursor.fetchone()[0] > 0:
                return

    # Create new or reset database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop existing tables if re-initializing
    tables = ["order_items", "orders", "products", "categories", "customers", "employees", "sales"]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    # 1. Categories
    cursor.execute("""
    CREATE TABLE categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT
    );
    """)

    cursor.executemany("""
    INSERT INTO categories (name, description) VALUES (?, ?)
    """, [
        ("Electronics", "Laptops, phones, audio gear, and gadgets"),
        ("Furniture", "Office desks, chairs, and ergonomic storage"),
        ("Clothing", "Apparel, sportswear, and fashion accessories"),
        ("Books", "Technical, educational, and fiction literature"),
        ("Home Appliances", "Smart kitchen and cleaning devices")
    ])

    # 2. Products
    cursor.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category_id INTEGER REFERENCES categories(id),
        price REAL NOT NULL,
        stock_quantity INTEGER NOT NULL DEFAULT 0,
        rating REAL DEFAULT 4.5
    );
    """)

    cursor.executemany("""
    INSERT INTO products (name, category_id, price, stock_quantity, rating) VALUES (?, ?, ?, ?, ?)
    """, [
        ("ProBook Laptop 15", 1, 1299.99, 45, 4.8),
        ("Noise-Cancelling Headphones", 1, 249.50, 120, 4.7),
        ("Wireless Mechanical Keyboard", 1, 119.00, 85, 4.6),
        ("Ultra-Wide 4K Monitor 32-inch", 1, 599.00, 30, 4.9),
        ("Ergonomic Mesh Chair", 2, 349.99, 25, 4.6),
        ("Standing Electric Desk", 2, 499.00, 18, 4.8),
        ("Leather Office Sofa", 2, 750.00, 10, 4.4),
        ("Merino Wool Winter Jacket", 3, 180.00, 60, 4.5),
        ("Running Breathable Sneakers", 3, 95.00, 140, 4.3),
        ("Classic Cotton Shirt", 3, 45.00, 200, 4.2),
        ("Designing Data-Intensive Applications", 4, 49.99, 150, 4.9),
        ("Deep Learning with Python", 4, 42.50, 90, 4.8),
        ("Modern Robotic Vacuum Cleaner", 5, 299.00, 40, 4.6),
        ("Smart Espresso Machine", 5, 399.00, 22, 4.7)
    ])

    # 3. Customers
    cursor.execute("""
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        city TEXT NOT NULL,
        country TEXT NOT NULL,
        balance REAL DEFAULT 0.0,
        created_at TEXT NOT NULL DEFAULT (date('now'))
    );
    """)

    cursor.executemany("""
    INSERT INTO customers (name, email, city, country, balance, created_at) VALUES (?, ?, ?, ?, ?, ?)
    """, [
        ("Ahmed Mansour", "ahmed.m@example.com", "Cairo", "Egypt", 1250.0, "2024-01-15"),
        ("Sara Al-Otaibi", "sara.o@example.com", "Riyadh", "Saudi Arabia", 3400.5, "2024-02-10"),
        ("Omar Khaled", "omar.k@example.com", "Alexandria", "Egypt", 420.0, "2024-03-01"),
        ("Fatima Al-Zahra", "fatima.z@example.com", "Dubai", "UAE", 5200.0, "2024-01-20"),
        ("John Smith", "john.s@example.com", "London", "UK", 890.0, "2024-02-18"),
        ("Elena Rossi", "elena.r@example.com", "Rome", "Italy", 150.0, "2024-03-12"),
        ("Youssef Hassan", "youssef.h@example.com", "Giza", "Egypt", 2100.0, "2024-02-25"),
        ("Nour El-Din", "nour.d@example.com", "Amman", "Jordan", 750.0, "2024-03-15"),
        ("Tariq Mahmoud", "tariq.m@example.com", "Jeddah", "Saudi Arabia", 1950.0, "2024-01-28"),
        ("Layla Saeed", "layla.s@example.com", "Cairo", "Egypt", 310.0, "2024-03-22")
    ])

    # 4. Employees
    cursor.execute("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        role TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL
    );
    """)

    cursor.executemany("""
    INSERT INTO employees (name, department, role, salary, hire_date) VALUES (?, ?, ?, ?, ?)
    """, [
        ("Karim Adel", "Engineering", "Lead AI Engineer", 8500.0, "2022-03-15"),
        ("Mona Sherif", "Engineering", "Full Stack Developer", 6200.0, "2023-01-10"),
        ("Zaid Qasim", "Engineering", "DevOps Specialist", 6800.0, "2022-09-01"),
        ("Hala Farouk", "Sales", "Sales Director", 7900.0, "2021-06-15"),
        ("Bassem Nader", "Sales", "Senior Account Executive", 5400.0, "2023-04-01"),
        ("Reem Mostafa", "Marketing", "Marketing Manager", 6100.0, "2022-11-20"),
        ("Ali Kamal", "Support", "Customer Success Lead", 4300.0, "2023-08-15"),
        ("Dina Samir", "Human Resources", "HR Generalist", 4800.0, "2022-05-10")
    ])

    # 5. Orders
    cursor.execute("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER REFERENCES customers(id),
        order_date TEXT NOT NULL,
        total_amount REAL NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('Delivered', 'Processing', 'Shipped', 'Cancelled'))
    );
    """)

    cursor.executemany("""
    INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES (?, ?, ?, ?)
    """, [
        (1, "2024-03-01", 1549.49, "Delivered"),
        (2, "2024-03-03", 599.00, "Delivered"),
        (3, "2024-03-05", 275.00, "Shipped"),
        (4, "2024-03-10", 2149.98, "Delivered"),
        (5, "2024-03-12", 119.00, "Processing"),
        (6, "2024-03-15", 399.00, "Delivered"),
        (7, "2024-03-18", 1299.99, "Processing"),
        (8, "2024-03-20", 92.49, "Delivered"),
        (9, "2024-03-22", 849.99, "Shipped"),
        (10, "2024-03-25", 249.50, "Delivered"),
        (1, "2024-04-01", 349.99, "Processing"),
        (4, "2024-04-03", 698.00, "Delivered")
    ])

    # 6. Order Items
    cursor.execute("""
    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER REFERENCES orders(id),
        product_id INTEGER REFERENCES products(id),
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL
    );
    """)

    cursor.executemany("""
    INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)
    """, [
        (1, 1, 1, 1299.99),
        (1, 2, 1, 249.50),
        (2, 4, 1, 599.00),
        (3, 8, 1, 180.00),
        (3, 9, 1, 95.00),
        (4, 1, 1, 1299.99),
        (4, 6, 1, 499.00),
        (4, 3, 2, 119.00),
        (5, 3, 1, 119.00),
        (6, 14, 1, 399.00),
        (7, 1, 1, 1299.99),
        (8, 11, 1, 49.99),
        (8, 12, 1, 42.50),
        (9, 6, 1, 499.00),
        (9, 5, 1, 349.99),
        (10, 2, 1, 249.50),
        (11, 5, 1, 349.99),
        (12, 13, 1, 299.00),
        (12, 14, 1, 399.00)
    ])

    # 7. Sales summary table (for quick analytical queries)
    cursor.execute("""
    CREATE TABLE sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_category TEXT NOT NULL,
        amount REAL NOT NULL,
        sale_date TEXT NOT NULL,
        region TEXT NOT NULL
    );
    """)

    cursor.executemany("""
    INSERT INTO sales (product_category, amount, sale_date, region) VALUES (?, ?, ?, ?)
    """, [
        ("Electronics", 1299.99, "2024-03-01", "North America"),
        ("Electronics", 249.50, "2024-03-01", "Middle East"),
        ("Furniture", 599.00, "2024-03-03", "Middle East"),
        ("Clothing", 275.00, "2024-03-05", "Europe"),
        ("Electronics", 2149.98, "2024-03-10", "Middle East"),
        ("Furniture", 499.00, "2024-03-15", "North America"),
        ("Home Appliances", 399.00, "2024-03-18", "Europe"),
        ("Books", 92.49, "2024-03-20", "Middle East"),
        ("Furniture", 849.99, "2024-03-22", "Middle East"),
        ("Electronics", 249.50, "2024-03-25", "Middle East"),
        ("Home Appliances", 698.00, "2024-04-03", "Europe")
    ])

    conn.commit()
    conn.close()


def get_sqlite_schema(db_path: str = DB_PATH) -> str:
    """
    Extracts the full DDL CREATE TABLE statements for all active tables in the SQLite database.
    This serves as prompt grounding for the Text-to-SQL LLM engine.
    """
    init_database(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' 
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name ASC;
        """)
        rows = cursor.fetchall()
        schema_statements = [row[0].strip() for row in rows if row and row[0]]
        return "\n\n".join(schema_statements)


def get_database_tables_info(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Returns table names and their column definitions for UI inspection."""
    init_database(db_path)
    tables = []
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name ASC")
        for table_name, ddl in cursor.fetchall():
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [
                {
                    "cid": col[0],
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "pk": bool(col[5])
                }
                for col in cursor.fetchall()
            ]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            tables.append({
                "table_name": table_name,
                "columns": columns,
                "row_count": count,
                "ddl": ddl
            })
    return tables


def execute_sqlite_query(sql_query: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Executes a SQL query against the SQLite database and returns structured table data.
    
    Args:
        sql_query: The SQL string (SELECT, INSERT, UPDATE, etc.)
        db_path: Path to the SQLite database file
        
    Returns:
        Dict with success, columns, rows (2D array), row_count, execution_time_ms, and optional error.
    """
    init_database(db_path)
    clean_sql = sql_query.strip()
    if not clean_sql:
        return {
            "success": False,
            "query": clean_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "execution_time_ms": 0.0,
            "error": "Query cannot be empty."
        }

    start_time = time.perf_counter()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(clean_sql)

            if cursor.description:
                # Query returned rows (e.g. SELECT, PRAGMA)
                columns = [desc[0] for desc in cursor.description]
                raw_rows = cursor.fetchall()
                # Format each row item cleanly (e.g. serialize floats nicely, keep strings)
                formatted_rows = [list(row) for row in raw_rows]
                row_count = len(formatted_rows)
            else:
                # DDL or modification statement (INSERT, UPDATE, DELETE, etc.)
                conn.commit()
                columns = ["status"]
                formatted_rows = [["Operation executed successfully."]]
                row_count = cursor.rowcount if cursor.rowcount >= 0 else 1

            execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return {
                "success": True,
                "query": clean_sql,
                "columns": columns,
                "rows": formatted_rows,
                "row_count": row_count,
                "execution_time_ms": execution_time_ms,
                "error": None
            }
    except sqlite3.Error as e:
        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "success": False,
            "query": clean_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "execution_time_ms": execution_time_ms,
            "error": f"SQLite Execution Error: {str(e)}"
        }
    except Exception as e:
        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "success": False,
            "query": clean_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "execution_time_ms": execution_time_ms,
            "error": f"Unexpected Error: {str(e)}"
        }
