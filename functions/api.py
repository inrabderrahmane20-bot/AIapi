import json
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from your main API file
try:
    from api import app
    HAS_API = True
except ImportError:
    HAS_API = False
    print("Warning: Could not import from api.py")

# For Netlify Lambda functions
def handler(event, context):
    """Netlify Lambda handler"""
    
    # If we don't have the API module, return a simple response
    if not HAS_API:
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'message': 'API is being initialized',
                'status': 'loading',
                'endpoints': [
                    '/api/cities',
                    '/api/cities/{name}',
                    '/api/search',
                    '/api/health'
                ]
            })
        }
    
    # Import Flask app components
    from flask import Request
    from werkzeug.datastructures import Headers
    import io
    
    # Parse the event
    path = event.get('path', '')
    http_method = event.get('httpMethod', 'GET')
    headers = Headers(event.get('headers', {}))
    query_string = event.get('queryStringParameters', {}) or {}
    body = event.get('body', '')
    
    # Remove the function path prefix
    if path.startswith('/.netlify/functions/api'):
        path = path[len('/.netlify/functions/api'):]
    
    # Handle empty path
    if path == '':
        path = '/'
    
    # Create WSGI environment
    environ = {
        'REQUEST_METHOD': http_method,
        'PATH_INFO': path,
        'QUERY_STRING': '&'.join([f'{k}={v}' for k, v in query_string.items()]),
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '443',
        'wsgi.url_scheme': 'https',
        'wsgi.input': io.BytesIO(body.encode() if body else b''),
        'wsgi.errors': sys.stderr,
        'wsgi.multithread': False,
        'wsgi.multiprocess': False,
        'wsgi.version': (1, 0),
    }
    
    # Add headers
    for key, value in headers.items():
        environ_key = f'HTTP_{key.upper().replace("-", "_")}'
        environ[environ_key] = value
    
    # Setup response
    response = {}
    response_body_chunks = []
    
    def start_response(status, response_headers, exc_info=None):
        response['status'] = status
        response['headers'] = dict(response_headers)
        return response_body_chunks.append
    
    # Call the Flask app
    try:
        app_iter = app.wsgi_app(environ, start_response)
        for chunk in app_iter:
            if chunk:
                response_body_chunks.append(chunk)
        
        if hasattr(app_iter, 'close'):
            app_iter.close()
    except Exception as e:
        print(f"Error in handler: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }
    
    # Combine response body
    response_body = b''.join(response_body_chunks)
    
    # Convert headers for API Gateway
    headers_dict = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
    }
    
    # Add any headers from Flask response
    if 'headers' in response:
        headers_dict.update(response['headers'])
    
    return {
        'statusCode': int(response.get('status', '200 OK').split()[0]),
        'headers': headers_dict,
        'body': response_body.decode('utf-8')
    }