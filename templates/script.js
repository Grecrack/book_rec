document.getElementById('user-form').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form's default submission behavior

    var user_id = document.getElementById('user-id').value;

    fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            'user_id': user_id,
        }),
    })
    .then(response => response.json())
    .then(data => {
        // Handle the response data...
        console.log(data);
    });
});
