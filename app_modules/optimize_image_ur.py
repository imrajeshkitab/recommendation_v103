def optimize_image_url(url: str, width: int = 380, resize: str = "contain") -> str:
    """
    Optimize Supabase image URL using the render/image endpoint with transformation parameters.
    """
    if not url:
        return url
    
    # Only optimize Supabase URLs
    if 'supabase.co' in url:
        # Replace /object with /render/image for image transformation API
        optimized_url = url.replace('/storage/v1/object/', '/storage/v1/render/image/')
        
        # Add transformation parameters
        separator = '&' if '?' in optimized_url else '?'
        return f"{optimized_url}{separator}width={width}&resize={resize}"
    
    return url
