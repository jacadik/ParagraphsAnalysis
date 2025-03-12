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
                !e.target.closest('a') && !e.target.closest('button')) {
                window.location.href = this.dataset.href;
            }
        });
    });
});
