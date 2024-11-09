from great_tables import TableBuilder, Table, Column, ColumnSpanner, Row, Cell

# Create the table
table = TableBuilder() \
    .add_table(Table(
        columns=[
            Column(width=5),
            Column(width=5),
            Column(width=5),
            Column(width=5)
        ],
        rows=[
            Row(cells=[
                Cell(content="Features", rowspan=2, stub=True),
                Cell(content="Method A", colspan=3, column_spanner=True),
                Cell(content="Method B", colspan=3, column_spanner=True),
                Cell(content="Method C", colspan=3, column_spanner=True),
            ]),
            Row(cells=[
                Cell(content="Min"),
                Cell(content="Mean"),
                Cell(content="Max"),
                Cell(content="Min"),
                Cell(content="Mean"),
                Cell(content="Max"),
                Cell(content="Min"),
                Cell(content="Mean"),
                Cell(content="Max"),
            ]),
            Row(cells=[
                Cell(content="Min"),
                Cell(content="Mean"),
                Cell(content="Max"),
                Cell(content="Min"),
                Cell(content="Mean"),
                Cell(content="Max"),
                Cell(content="Min"),
                Cell(content="Mean"),
                Cell(content="Max"),
            ])
        ]
    ))

# Render the table
print(table.render())

