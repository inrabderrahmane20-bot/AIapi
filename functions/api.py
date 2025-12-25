import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app
import json

# For Netlify functions, we need a handler
def handler(event, context):
    from flask import Request, Response
    from werkzeug.datastructures import Headers
    import io
    
    # Parse the event
    path = event.get('path', '')
    http_method = event.get('httpMethod', 'GET')
    headers = Headers(event.get('headers', {}))
    query_string = event.get('queryStringParameters', {}) or {}
    body = event.get('body', '')
    
    # Create a fake WSGI environment
    environ = {
        'REQUEST_METHOD': http_method,
        'PATH_INFO': path.replace('/.netlify/functions/api', ''),
        'QUERY_STRING': '&'.join([f'{k}={v}' for k, v in query_string.items()]),
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '443',
        'wsgi.url_scheme': 'https',
        'wsgi.input': io.BytesIO(body.encode() if body else b''),
        'wsgi.errors': sys.stderr,
        'wsgi.multithread': True,
        'wsgi.multiprocess': False,
        'wsgi.version': (1, 0),
        'HTTP_X_FORWARDED_PROTO': 'https'
    }
    
    # Update with headers
    for key, value in headers.items():
        environ[f'HTTP_{key.upper().replace("-", "_")}'] = value
    
    # Create response
    response = {}
    
    def start_response(status, response_headers):
        response['status'] = status
        response['headers'] = dict(response_headers)
    
    # Call the app
    app_iter = app.wsgi_app(environ, start_response)
    
    # Get the response body
    response_body = b''.join(app_iter)
    
    # Return the response
    return {
        'statusCode': int(response['status'].split()[0]),
        'headers': response['headers'],
        'body': response_body.decode('utf-8')
    }