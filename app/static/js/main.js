document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    const preview = document.getElementById('preview');
    
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.addEventListener('load', function() {
                preview.src = reader.result;
                preview.classList.remove('d-none');
            });
            
            reader.readAsDataURL(file);
        }
    });
});