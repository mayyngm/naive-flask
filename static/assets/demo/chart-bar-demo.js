// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("myBarChart1");
var myLineChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["Negative", "Positive"],
    datasets: [{
      label: "Revenue",
      backgroundColor: ['#ef4444','#22c55e'],
      borderColor: ['#ef4444','#22c55e'],
      data: [10, 10],
    }],
  },
  options: {
    scales: {
      xAxes: [{
        time: {
          unit: 'month'
        },
        gridLines: {
          display: false
        },
        ticks: {
          maxTicksLimit: 6
        }
      }],
      yAxes: [{
        ticks: {
          min: 0,
          max: 100,
          maxTicksLimit: 5
        },
        gridLines: {
          display: true
        }
      }],
    },
    legend: {
      display: false
    },
    title:{
      display: true,
      text: 'Sinovac'
    }
  }
});



var ctx = document.getElementById("myBarChart2");
var myLineChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["Negative", "Positive"],
    datasets: [{
      label: "Revenue",
      backgroundColor: ['#ef4444','#22c55e'],
      borderColor: ['#ef4444','#22c55e'],
      data: [10, 10],
    }],
  },
  options: {
    scales: {
      xAxes: [{
        time: {
          unit: 'month'
        },
        gridLines: {
          display: false
        },
        ticks: {
          maxTicksLimit: 6
        }
      }],
      yAxes: [{
        ticks: {
          min: 0,
          max: 100,
          maxTicksLimit: 5
        },
        gridLines: {
          display: true
        }
      }],
    },
    legend: {
      display: false
    },
    title:{
      display: true,
      text: 'Astrazeneca'
    }
  }
});


var ctx = document.getElementById("myBarChart3");
var myLineChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["Negative", "Positive"],
    datasets: [{
      label: "Revenue",
      backgroundColor: ['#ef4444','#22c55e'],
      borderColor: ['#ef4444','#22c55e'],
      data: [10, 10],
    }],
  },
  options: {
    scales: {
      xAxes: [{
        time: {
          unit: 'month'
        },
        gridLines: {
          display: false
        },
        ticks: {
          maxTicksLimit: 6
        }
      }],
      yAxes: [{
        ticks: {
          min: 0,
          max: 100,
          maxTicksLimit: 5
        },
        gridLines: {
          display: true
        }
      }],
    },
    legend: {
      display: false
    },
    title:{
      display: true,
      text: 'Pfizer'
    }
  }
});


var ctx = document.getElementById("myBarChart4");
var myLineChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["Negative", "Positive"],
    datasets: [{
      label: "Revenue",
      backgroundColor: ['#ef4444','#22c55e'],
      borderColor: ['#ef4444','#22c55e'],
      data: [10, 10],
    }],
  },
  options: {
    scales: {
      xAxes: [{
        time: {
          unit: 'month'
        },
        gridLines: {
          display: false
        },
        ticks: {
          maxTicksLimit: 6
        }
      }],
      yAxes: [{
        ticks: {
          min: 0,
          max: 100,
          maxTicksLimit: 5
        },
        gridLines: {
          display: true
        }
      }],
    },
    legend: {
      display: false
    },
    title:{
      display: true,
      text: 'Moderna'
    }
  }
});