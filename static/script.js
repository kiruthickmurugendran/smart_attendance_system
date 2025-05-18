// Global variables
let videoStream = null;

// Show selected section
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.style.display = 'none';
    });
    document.getElementById(sectionId).style.display = 'block';
    loadSectionData(sectionId);
}

// Load data for the current section
function loadSectionData(sectionId) {
    switch (sectionId) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'attendance':
            loadAttendance();
            break;
        case 'absent':
            loadAbsentRequests();
            break;
        case 'stats':
            loadStatistics();
            break;
        case 'capture':
            setupVideoFeed();
            break;
    }
}

// Dashboard
function loadDashboard() {
    fetch('/api/attendance/stats')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            let defaultersHtml = '<table class="table table-hover"><thead><tr><th>Name</th><th>Roll No</th><th>Percentage</th></tr></thead><tbody>';

            if (data.defaulters && data.defaulters.length > 0) {
                data.defaulters.forEach(defaulter => {
                    defaultersHtml += `<tr>
                        <td>${defaulter.name}</td>
                        <td>${defaulter.roll_number}</td>
                        <td><div class="progress"><div class="progress-bar bg-danger" style="width: ${defaulter.percentage}%">${defaulter.percentage}%</div></div></td>
                    </tr>`;
                });
            } else {
                defaultersHtml += '<tr><td colspan="3" class="text-center">No defaulters found</td></tr>';
            }

            defaultersHtml += '</tbody></table>';
            document.getElementById('defaultersList').innerHTML = defaultersHtml;

            const today = new Date().toISOString().split('T')[0];
            fetch(`/api/attendance?date=${today}`)
                .then(response => {
                    if (!response.ok) throw new Error('Network response was not ok');
                    return response.json();
                })
                .then(attendance => {
                    let attendanceHtml = '<table class="table table-hover"><thead><tr><th>Name</th><th>Roll No</th><th>Status</th></tr></thead><tbody>';

                    if (attendance && attendance.length > 0) {
                        attendance.forEach(record => {
                            attendanceHtml += `<tr>
                                <td>${record.name}</td>
                                <td>${record.roll_number}</td>
                                <td><span class="badge ${record.present ? 'bg-success' : 'bg-danger'}">${record.present ? 'Present' : 'Absent'}</span></td>
                            </tr>`;
                        });
                    } else {
                        attendanceHtml += '<tr><td colspan="3" class="text-center">No attendance records for today</td></tr>';
                    }

                    attendanceHtml += '</tbody></table>';
                    document.getElementById('todayAttendance').innerHTML = attendanceHtml;
                })
                .catch(error => {
                    console.error('Error loading today attendance:', error);
                    document.getElementById('todayAttendance').innerHTML = '<div class="alert alert-danger">Error loading today\'s attendance</div>';
                });
        })
        .catch(error => {
            console.error('Error loading dashboard:', error);
            document.getElementById('defaultersList').innerHTML = '<div class="alert alert-danger">Error loading defaulters list</div>';
            document.getElementById('todayAttendance').innerHTML = '<div class="alert alert-danger">Error loading today\'s attendance</div>';
        });
}

// Attendance
function loadAttendance() {
    const date = document.getElementById('attendanceDate')?.value;
    const url = date ? `/api/attendance?date=${date}` : '/api/attendance';

    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            let html = '<table class="table table-hover"><thead><tr><th>Date</th><th>Roll No</th><th>Name</th><th>Status</th></tr></thead><tbody>';

            if (data && data.length > 0) {
                data.forEach(record => {
                    html += `<tr>
                        <td>${record.date}</td>
                        <td>${record.roll_number}</td>
                        <td>${record.name}</td>
                        <td><span class="badge ${record.present ? 'bg-success' : 'bg-danger'}">${record.present ? 'Present' : 'Absent'}</span></td>
                    </tr>`;
                });
            } else {
                html += '<tr><td colspan="4" class="text-center">No attendance records found</td></tr>';
            }

            html += '</tbody></table>';
            document.getElementById('attendanceRecords').innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading attendance:', error);
            document.getElementById('attendanceRecords').innerHTML = '<div class="alert alert-danger">Error loading attendance records</div>';
        });
}

// Absent Requests
function loadAbsentRequests() {
    fetch('/api/absent')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            let html = '<table class="table table-hover"><thead><tr><th>Date</th><th>Roll No</th><th>Name</th><th>Department</th><th>Reason</th></tr></thead><tbody>';

            if (data && data.length > 0) {
                data.forEach(request => {
                    html += `<tr>
                        <td>${request.date}</td>
                        <td>${request.roll_number}</td>
                        <td>${request.name}</td>
                        <td>${request.department}</td>
                        <td>${request.reason}</td>
                    </tr>`;
                });
            } else {
                html += '<tr><td colspan="5" class="text-center">No absent requests found</td></tr>';
            }

            html += '</tbody></table>';
            document.getElementById('absentRequests').innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading absent requests:', error);
            document.getElementById('absentRequests').innerHTML = '<div class="alert alert-danger">Error loading absent requests</div>';
        });
}

// Submit absent request
function submitAbsentRequest() {
    const rollNumber = document.getElementById('absentRollNumber').value;
    const department = document.getElementById('absentDepartment').value;
    const reason = document.getElementById('absentReason').value;

    if (!rollNumber || !department || !reason) {
        alert('Please fill all fields');
        return;
    }

    fetch('/api/absent/request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roll_number: rollNumber, department, reason })
    })
        .then(response => {
            if (!response.ok) return response.json().then(err => { throw err; });
            return response.json();
        })
        .then(data => {
            alert(data.message);
            document.getElementById('absentForm').reset();
            bootstrap.Modal.getInstance(document.getElementById('absentModal')).hide();
            loadAbsentRequests();
        })
        .catch(error => {
            console.error('Error submitting absent request:', error);
            alert(error.detail || 'Error submitting absent request');
        });
}

// ✅ Improved: Setup MJPEG video stream
function setupVideoFeed() {
    const startBtn = document.getElementById('startCapture');
    const stopBtn = document.getElementById('stopCapture');
    const videoFeed = document.getElementById('videoFeed');
    const recognitionLog = document.getElementById('recognitionLog');

    startBtn.addEventListener('click', () => {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        recognitionLog.innerHTML = 'Starting camera...<br>';

        // Ensure it's visible
        videoFeed.style.display = 'block';

        // Set stream
        videoFeed.src = '/video_feed';

        videoFeed.onerror = () => {
            recognitionLog.innerHTML += '❌ Error loading video stream. Make sure backend is running.<br>';
            stopBtn.click();
        };
    });

    stopBtn.addEventListener('click', () => {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        videoFeed.src = '';
        recognitionLog.innerHTML += 'Camera stopped<br>';
        videoFeed.style.display = 'none';
    });
}

// Register student
document.getElementById('registerForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const rollNumber = document.getElementById('rollNumber').value;
    const name = document.getElementById('studentName').value;
    const department = document.getElementById('department').value;
    const faceImage = document.getElementById('faceImage').files[0];

    if (!faceImage) {
        alert('Please capture a face image');
        return;
    }

    const formData = new FormData();
    formData.append('roll_number', rollNumber);
    formData.append('name', name);
    formData.append('department', department);
    formData.append('image', faceImage);

    fetch('/api/face/register', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) return response.json().then(err => { throw err; });
            return response.json();
        })
        .then(data => {
            alert(data.message);
            document.getElementById('registerForm').reset();
        })
        .catch(error => {
            console.error('Error registering student:', error);
            alert(error.detail || 'Error registering student');
        });
});

// Stats
function loadStatistics() {
    fetch('/api/attendance/stats')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            if (data.plot) {
                document.getElementById('attendancePlot').innerHTML = `<img src="data:image/png;base64,${data.plot}" class="img-fluid">`;
            } else {
                document.getElementById('attendancePlot').innerHTML = '<div class="alert alert-warning">No attendance plot available</div>';
            }

            let summaryHtml = '<table class="table table-hover"><thead><tr><th>Name</th><th>Roll No</th><th>Present Days</th><th>Total Days</th><th>Percentage</th></tr></thead><tbody>';

            if (data.stats && data.stats.length > 0) {
                data.stats.forEach(stat => {
                    const progressClass = stat.percentage < 75 ? 'bg-danger' : 'bg-success';
                    summaryHtml += `<tr>
                        <td>${stat.name}</td>
                        <td>${stat.roll_number}</td>
                        <td>${stat.present_days}</td>
                        <td>${stat.total_days}</td>
                        <td><div class="progress"><div class="progress-bar ${progressClass}" style="width: ${stat.percentage}%">${stat.percentage}%</div></div></td>
                    </tr>`;
                });
            } else {
                summaryHtml += '<tr><td colspan="5" class="text-center">No attendance statistics available</td></tr>';
            }

            summaryHtml += '</tbody></table>';
            document.getElementById('attendanceSummary').innerHTML = summaryHtml;
        })
        .catch(error => {
            console.error('Error loading statistics:', error);
            document.getElementById('attendancePlot').innerHTML = '<div class="alert alert-danger">Error loading attendance plot</div>';
            document.getElementById('attendanceSummary').innerHTML = '<div class="alert alert-danger">Error loading attendance summary</div>';
        });
}

// On page load
document.addEventListener('DOMContentLoaded', function () {
    showSection('dashboard');
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('attendanceDate').value = today;

    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
