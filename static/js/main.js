// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    // Add confirmation for potentially destructive actions
    const confirmActions = document.querySelectorAll('[data-confirm]');
    confirmActions.forEach(element => {
        element.addEventListener('click', function(e) {
            const message = this.getAttribute('data-confirm');
            if (!confirm(message)) {
                e.preventDefault();
            }
        });
    });
    
    // Add tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Table row click to navigate (document list)
    const clickableRows = document.querySelectorAll('tr[data-href]');
    clickableRows.forEach(row => {
        row.style.cursor = 'pointer';
        row.addEventListener('click', function(e) {
            // Don't trigger if they clicked a button or link
            if (e.target.tagName !== 'A' && e.target.tagName !== 'BUTTON' && 
                !e.target.closest('a') && !e.target.closest('button') &&
                !e.target.closest('.form-check') && e.target.type !== 'checkbox') {
                window.location.href = this.dataset.href;
            }
        });
    });
    
    // Document deletion confirmation
    const deleteButtons = document.querySelectorAll('.delete-document-btn');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const docName = this.getAttribute('data-document-name');
            if (!confirm(`Are you sure you want to delete "${docName}"? This cannot be undone.`)) {
                e.preventDefault();
            }
        });
    });
    
    // Select/deselect all documents
    const selectAllCheckbox = document.getElementById('selectAll');
    if (selectAllCheckbox) {
        const documentCheckboxes = document.querySelectorAll('.document-checkbox');
        
        selectAllCheckbox.addEventListener('change', function() {
            documentCheckboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
            });
        });
        
        // Update "select all" checkbox when individual checkboxes change
        documentCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                const allChecked = Array.from(documentCheckboxes).every(cb => cb.checked);
                selectAllCheckbox.checked = allChecked;
            });
        });
    }
    
    // Bulk action form submission validation
    const bulkActionForm = document.getElementById('bulkActionForm');
    if (bulkActionForm) {
        bulkActionForm.addEventListener('submit', function(e) {
            const selectedDocs = document.querySelectorAll('.document-checkbox:checked');
            if (selectedDocs.length === 0) {
                e.preventDefault();
                alert('Please select at least one document.');
            }
        });
    }
});
