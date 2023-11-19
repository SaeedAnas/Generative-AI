import argparse

from asifbot import config
from asifbot.core.db import Postgres, Qdrant
from asifbot.schema.sql import TABLES

def create_command(args):
    print("Creating tables")
    Postgres().create_tables()
    Qdrant().create_collection()
        
def drop_command(args):
    # Logic for the drop command
    print("Dropping tables")
    Postgres().drop_tables()
    Qdrant().delete_collection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    create_parser = subparsers.add_parser('create')
    create_parser.set_defaults(func=create_command)

    drop_parser = subparsers.add_parser('drop')
    drop_parser.set_defaults(func=drop_command)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()