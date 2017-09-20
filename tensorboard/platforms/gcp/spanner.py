# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cloud Spanner support.

This module adds functionality to use Cloud Spanner as the backing database for
TensorBoard. TensorBoard is designed to use a PEP 249 db. Cloud Spanner doesn't
have a PEP 249 compliant DB.
See:
https://github.com/GoogleCloudPlatform/google-cloud-python/wiki/Feature-Backlog.

A PEP 249 API for Cloud Spanner is blocked by lack of DML support for Cloud
Spanner. This shouldn't block supporting Cloud Spanner with TensorBoard because
TensorBoard is largely read only.

WARNING: This module is EXPERIMENTAL. It will not be considered stable
until data migration tools are put into place. Until that time, any
database created with this schema will need to be deleted as new updates
occur.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import re
import six
from google.cloud import spanner  # pylint: disable=import-error
from google.gax import errors
from tensorboard import db
from tensorboard import schema


def to_spanner_type(column_type):
  """Return the Cloud Spanner type corresponding to the supplied type.

  Args:
    column_type: Instance of ColumnType.

  Returns:
    string identify the spanner column type.
  """
  if isinstance(column_type, schema.BoolColumnType):
    return "BOOL"

  if isinstance(column_type, schema.BytesColumnType):
    if column_type.length:
      return 'BYTES({0})'.format(column_type.length)
    else:
      return 'BYTES(MAX)'

  if isinstance(column_type, schema.Int64ColumnType):
    return 'INT64'

  if isinstance(column_type, schema.StringColumnType):
    if column_type.length:
      return 'STRING({0})'.format(column_type.length)
    else:
      return 'STRING(MAX)'

  raise ValueError(
      '{0} is not a support ColumnType'.format(column_type.__class__))


def to_spanner_ddl(spec):
  """Convert a TableSchema object to a spanner DDL statement.

  Args:
    spec: TableSchema object representing the schema for the table.

  Returns:
    ddl statement to create the table.

  : type spec: TableSchema
  : rtype : str
  """
  # TODO(jlewi): Add support for not null modifier.
  if isinstance(spec, schema.TableSchema):
    columns = []
    for c in spec.columns:
      s = '{0} {1}'.format(c.name, to_spanner_type(c.value_type))
      columns.append(s)
    columns = ', '.join(columns)
    keys = ', '.join(spec.keys)
    ddl = 'CREATE TABLE {name} ({columns}) PRIMARY KEY ({key_fields})'.format(
        name=spec.name, columns=columns, key_fields=keys)

  elif isinstance(spec, schema.IndexSchema):
    ddl = ('CREATE UNIQUE INDEX {name} ON {table} ({columns})').format(
        name=spec.name, table=spec.table,
        columns=', '.join(spec.columns))
  return ddl


class CloudSpannerConnection(object):
  """Connection to Cloud Spanner database.
  """

  def __init__(self, project, instance, database):
    """Create a connection to a Cloud Spanner Database

    Args:
      project: The project that owns the DB.
      instance: The name of the instance to use.
      database: The name of the database.
    """
    # TODO(jlewi): Should we take client as an argument?
    self.client = spanner.Client(project=project)
    self.instance_id = instance
    self.database_id = database
    self._instance = None
    self._database = None

  def cursor(self):
    """Construct a cursor for Cloud Spanner."""
    # TODO(jlewi): Is constructing a db.Cursor with delegate set to
    # CloudSpannerCursor the right pattern?
    delegate = CloudSpannerCursor(
        self.client, self.database_id, self.instance_id)
    cursor = db.Cursor(self)
    cursor._delegate = delegate
    return cursor

  @property
  def database(self):
    """Return the CloudSpanner Database object.

    This method is not part of PEP249.
    """
    if not self._database:
      self._database = self.instance.database(self.database_id)
    return self._database

  @property
  def instance(self):
    """Return the CloudSpanner instance object.

    This method is not part of PEP249.
    """
    if not self._instance:
      self._instance = spanner.client.Instance(self.instance_id, self.client)
    return self._instance


class CloudSpannerCursor(object):
  """Cursor for Cloud Spanner.

  When executing an SQL select query, the cursor will load all rows into memory
  when execute as called as opposed to streaming the results based on calls to
  fetchone and fetchmany. This is a pretty naive implementation that could be
  inefficient when returning many rows. We should consider improving that in
  the future.
  """

  def __init__(self, client, database_id, instance_id):
    """ Create the Cursor.

    :param client: Spanner client.
    :param dabase_id: Database id.
    """
    self.client = client
    self.database_id = database_id
    self.instance_id = instance_id
    self.database = None
    # TODO(jlewi): Should we take in a CloudSpannerConnection and reuse the
    # instance and db associated with that connection?
    self.instance = spanner.client.Instance(instance_id, client)

    # Store results of an SQL query for use in cursor operation
    # rindex points to the position in _results of the next row to return.
    self._results = []
    self._rindex = 0
    self._descriptions = []

    # PEP 249 says that arraysize should be an attribute the controls batch
    # size for various operations e.g. when streaming the results from a query
    # we could fetch them in batches of arraysize.
    # TODO(jlewi): The value of 10 was a randomly picked number. I have no idea
    # what a sensible default would be.
    self.arraysize = 10

  def execute(self, sql, parameters=()):
    """Executes a single query.

    :type sql: str
    :type parameters: tuple[object]
    """
    # TODO(jlewi): Should we check that the DB exists and if not raise an error?
    # TODO(jlewi): What is the substitution syntax for parameters? Is this a PEP
    # 249 convention?
    # TODO(jlewi): According to db.Connection.execute execute shouldn't execute
    # until end of transaction so we may need to rethink how this works.
    # I just guessed that Python format would work.
    self.database = self.instance.database(self.database_id)

    parsed = parse_sql(sql, parameters)

    if not parsed:
      raise ValueError(
          'SQL query {} is not supported for Cloud Spanner.'.format(sql))

    if isinstance(parsed, InsertSQL):
      with self.database.batch() as batch:
        batch.insert(
            table=parsed.table,
            columns=parsed.columns,
            values=[parsed.values])

      return

    if isinstance(parsed, SelectSQL):
      session = self.database.session()
      session.create()
      results = session.execute_sql(parsed.sql)
      results.consume_all()
      self._results = results.rows
      self._rindex = 0

      # TODO(jlewi): We should support the type_code as well.
      # According to https://www.python.org/dev/peps/pep-0249/#cursor-attributes
      # name and type_code are the only two attributes required for a column
      # description. However, it wasn't clear from the spec what values we
      # should use for the type_code so I just it to None for now.
      self._descriptions = []
      for c in parsed.columns:
        self._descriptions.append([c, None, None, None, None, None, None])
      return

  def executemany(self, sql, seq_of_parameters=()):
    """Executes a single query many times.

    :type sql: str
    :type seq_of_parameters: list[tuple[object]]
    """
    for p in seq_of_parameters:
      self.execute(sql, p)

  def executescript(self, sql):
    """Executes a script of many queries.

    :type sql: str
    """
    raise NotImplementedError(
        'executescript is not a PEP249 method and is not currently '
        'supported for Cloud Spanner')

  def fetchone(self):
    """Returns next row in result set.

    :rtype: tuple[object]
    """
    if self._rindex < len(self._results):
      self._rindex += 1
      return self._results[self._rindex - 1]

  def fetchmany(self, size=None):
    """Returns next chunk of rows in result set.

    :type size: int
    """
    start_index = self._rindex
    if size is not None:
      end_index = start_index + size
    else:
      end_index = len(self._results)

    self._rindex = end_index
    return self._results[start_index:end_index]

  def fetchall(self):
    """Returns next row in result set.

    :rtype: tuple[object]
    """
    start_index = self._rindex
    end_index = len(self._results)
    self._rindex = end_index
    return self._results[start_index:end_index]

  @property
  def description(self):
    """Returns information about each column in result set.

    See: https://www.python.org/dev/peps/pep-0249/

    :rtype: list[tuple[str, int, int, int, int, int, bool]]
    """
    # First two columns are name and typecode and required.
    # The others are optional.
    return self._descriptions

  @property
  def rowcount(self):
    """Returns number of rows retrieved by last read query.

    :rtype: int
    """
    return len(self._results)

  @property
  def lastrowid(self):
    """Returns last row ID.

    :rtype: int
    """
    # According to PEP249 this should be set to None if the Database doesn't
    # support RowIDs.
    return None

  def close(self):
    """Closes resources associated with cursor."""
    # TODO(jlewi): Are there any spanner resources that should be released
    pass

  def __iter__(self):
    """Returns iterator over results of last read query.

    :rtype: types.GeneratorType[tuple[object]]
    """
    for row in self._results:
      yield row

  def nextset(self):
    """Raises NotImplementedError."""
    raise NotImplementedError('Cursor.nextset not supported')

  def callproc(self, procname, parameters=()):
    """Raises NotImplementedError."""
    raise NotImplementedError('Cursor.callproc not supported')

  def setinputsizes(self, sizes):
    """Raises NotImplementedError."""
    raise NotImplementedError('Cursor.setinputsizes not supported')

  def setoutputsize(self, size, column):
    """Raises NotImplementedError."""
    raise NotImplementedError('Cursor.setoutputsize not supported')


def create_database(client, instance_id, database_id):
  """Creates a Cloud Spanner Database for TensorBoard.

  Args:
    client: A cloud spanner client.
    instance_id: The id of the instance.
    database_id: The id of the database

  Raises:
    RetryError if there is a problem creating the tables or indexes.
  """
  ddl = [to_spanner_ddl(t) for t in schema.TABLES]
  ddl.extend([to_spanner_ddl(t) for t in schema.INDEXES])

  client = client
  instance = spanner.client.Instance(instance_id, client)

  database = instance.database(database_id, ddl)
  try:
    op = database.create()
    op.result()
  except errors.RetryError as e:
    logging.error("There was a problem creating the database. %s",
                  e.cause.details())
    raise


class InsertSQL(object):
  """Represent and InsertSQL statement."""

  def __init__(self, table, columns, values):
    self.table = table
    self.columns = columns
    self.values = values


# \s matches any whitespace
# For the values we allow any character inside the parantheses because
# string values could include any character.
INSERT_PATTERN = re.compile(
    r'\s*insert\s*into\s*([a-z0-9_]*)\s*\(([a-z0-9,_\s]*)\)\s*values\s*'
    r'\((.*)\)', flags=re.IGNORECASE)

SELECT_PATTERN = re.compile(r'\s*select.*', flags=re.IGNORECASE)


class SelectSQL(object):
  """Reprsent a Select SQL statement."""

  _PATTERN = re.compile(
      r'\s*select\s*([a-z0-9_,\s]*)from\s*([a-z0-9,_]*).*', flags=re.IGNORECASE)
  _COL_NAME_PATTERN = re.compile(r'\s*([a-zA-z0-9_]*)\s*.*')

  def __init__(self, sql):
    """Construct an SQL query."""
    self.sql = sql

    m = self._PATTERN.match(sql)
    if not m:
      raise ValueError(
          'Could not parse columns and table name from query: {0}'.format(sql))
    self.table = m.group(2)
    columns = m.group(1).split(',')
    self.columns = []
    for c in columns:
      m = self._COL_NAME_PATTERN.match(c)
      if not m:
        raise ValueError('Could not parse column name from: {0}'.format(c))
      self.columns.append(m.group(1))


def parse_sql(sql, parameters):
  """Parse an sql statement.

  Args:
    sql: An SQL statement.
    parameters: Parameters to substitute into the query.

  Returns:
    obj: InsertSQL or SelectSQL or UpdateSQL object containing the result.

  : type sql:str
  : type paramters: List[Object]
  : rtype: InsertSQL | SelectSQL | UpdateSQL | None
  """

  # Perform variable substitution
  sql = sql.replace("?", "{}")
  # Convert tuple to list so its modifiable.
  parameters = list(parameters)
  # strings need to be quoted
  for i, p in enumerate(parameters):
    if isinstance(p, six.string_types):
      parameters[i] = '"{0}"'.format(p)
  sql = sql.format(*parameters)

  m = INSERT_PATTERN.match(sql)
  if m:
    table = m.group(1)
    columns = [c.strip() for c in m.group(2).split(',')]
    values = [v.strip() for v in m.group(3).split(',')]
    return InsertSQL(table, columns, values)

  m = SELECT_PATTERN.match(sql)
  if m:
    return SelectSQL(sql)
